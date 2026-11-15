"""研究流水线编排：所有输出写入调用方给定的目录，不修改 fixtures。

输出围绕一个区分展开：
- 预测排序质量：排序器对日志选中项目的加权对数损失、命中与排名；
- 策略价值：在明确的目标策略动作概率下的 IPS/SNIPS、ESS、覆盖、按请求 bootstrap；
- 支持性：目标质量落在日志零支持动作上时，价值不可识别，只报下界并拒绝完整收益。
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import ope
from .baseline import train_popularity_baseline
from .data import (
    ItemCatalog,
    Impression,
    ItemVersion,
    Splits,
    check_item_references,
    deduplicate,
    load_catalog,
    load_fixture_dir,
    make_splits,
    parse_time,
    read_policies,
    split_overlap_report,
)
from .errors import InputContractError, SupportError
from .policy import (
    build_list_policy,
    decision_to_dict,
    resolve_availability,
)
from .ranker import (
    baseline_scores_for_request,
    cold_start_template,
    ranker_scores,
    tune_ranker,
)

HAND_EXPECTED = {"ips": 0.25, "snips": 1.0 / 3.0}
TOLERANCE = 1e-9


def _rng_factory(seed: int = 0):
    return lambda value: random.Random(value * 1000003 + seed)


def _decisions(impressions, catalog, score_fn, cfg, seed, max_exact_lists):
    return {
        imp.request_id: build_list_policy(
            imp, catalog, score_fn(imp),
            list_size=cfg["list_size"],
            max_per_category=cfg["max_per_category"],
            seed=seed,
            max_exact_lists=max_exact_lists,
        )
        for imp in impressions
    }


def _display_target(decisions):
    """部署策略：从展示列表均匀抽一项。π 在列表上求和为 1，不含未观测质量。"""
    return lambda imp: dict(decisions[imp.request_id].deterministic_draw_probs)


def _randomized_target(decisions):
    """随机列表策略：Σ_L w(L)/|L|。所有合法列表等长，π 求和为 1。"""
    return lambda imp: dict(decisions[imp.request_id].randomized_action_probs)


def run_hand_check(impressions, cfg, uniform_verified) -> dict:
    wanted = list(cfg["hand_calculation_requests"])
    by_id = {imp.request_id: imp for imp in impressions}
    missing = [rid for rid in wanted if rid not in by_id]
    if missing:
        raise InputContractError(f"手算请求缺失：{missing}")
    selected = [by_id[rid] for rid in wanted]
    manual = cfg["target_probs"]

    def target_for(imp):
        prob = manual.get(imp.request_id)
        if prob is None:
            raise InputContractError(f"policies.json 缺少 {imp.request_id} 的目标动作概率")
        # policies 只给出目标在被记录动作上的概率；在候选集均匀核验下，
        # 其余候选动作的 μ 已知且为正，不构成支持损失。
        support = {imp.selected_item: prob}
        remainder = max(0.0, 1.0 - prob)
        if remainder > 0:
            support[ope.CANDIDATE_REMAINDER] = remainder
        return support

    result = ope.evaluate_policy(
        selected, "reward_click", target_for,
        repeats=cfg["bootstrap_repeats"], seed=cfg["seed"],
        rng_factory=_rng_factory(cfg["seed"]),
        uniform_verified=uniform_verified, clip=None,
    )
    ips = result["unclipped"]["ips"]
    snips = result["unclipped"]["snips"]
    assertions = {
        "ips_expected": HAND_EXPECTED["ips"],
        "ips_actual": ips,
        "ips_ok": abs(ips - HAND_EXPECTED["ips"]) <= TOLERANCE,
        "snips_expected": HAND_EXPECTED["snips"],
        "snips_actual": snips,
        "snips_ok": snips is not None and abs(snips - HAND_EXPECTED["snips"]) <= TOLERANCE,
        "ess_expected": (0.5 + 1.0) ** 2 / (0.25 + 1.0),
        "ess_actual": result["unclipped"]["ess"],
        "ess_ok": abs(result["unclipped"]["ess"] - (0.5 + 1.0) ** 2 / (0.25 + 1.0)) <= 1e-9,
        "derivation": (
            "权重 = 目标概率/记录概率：0.25/0.5=0.5 与 1/1=1。"
            "IPS=(0.5*1+1*0)/2=0.25；SNIPS=(0.5*1+1*0)/(0.5+1)=1/3；"
            "ESS=(0.5+1)^2/(0.5^2+1^2)=1.8。"
        ),
    }
    result["assertions"] = assertions
    if not (assertions["ips_ok"] and assertions["snips_ok"] and assertions["ess_ok"]):
        raise SupportError(f"手算校验失败：{assertions}")
    result["interpretation"] = (
        "IPS/SNIPS 在 policies 给出的目标被记录动作概率下计算。其余候选动作的目标质量未逐项规定，"
        "但记录策略在候选集上均匀已核验，其 μ 为正；因此这可读作该候选集内目标的离线估计。"
    )
    return result


def _bounds(rows):
    return (min((i.requested_at for i in rows), default=None),
            max((i.requested_at for i in rows), default=None))


def run_time_leakage_audit(splits: Splits, catalog: ItemCatalog) -> dict:
    report = split_overlap_report(splits)
    violations = []
    for name, rows in splits.as_dict().items():
        for imp in rows:
            for item in imp.candidates:
                try:
                    snapshot = catalog.as_of(item, imp.requested_at)
                except Exception as exc:  # noqa: BLE001
                    violations.append({"split": name, "request_id": imp.request_id,
                                       "item": item, "problem": f"no_visible_version: {exc}"})
                    continue
                if snapshot.version_time and snapshot.version_time > imp.requested_at:
                    violations.append({"split": name, "request_id": imp.request_id,
                                       "item": item, "problem": "effective_after_request"})
                if snapshot.known_at and snapshot.known_at > imp.requested_at:
                    violations.append({"split": name, "request_id": imp.request_id,
                                       "item": item, "problem": "known_after_request"})
    tmin, tmax = _bounds(splits.train)
    vmin, vmax = _bounds(splits.validation)
    smin, smax = _bounds(splits.test)
    ordered = True
    if tmax and vmin and not tmax < vmin:
        ordered = False
    if vmax and smin and not vmax < smin:
        ordered = False
    # 负面演示：若错误地用“全历史最后版本”，闭店前的请求会被未来状态影响。
    counterfactual = []
    demo_time = parse_time("2026-10-19T12:00:00+08:00", "demo")
    correct = catalog.as_of("f", demo_time)
    latest = catalog.as_of("f", parse_time("2026-10-25T00:00:00+08:00", "demo"))
    counterfactual.append(
        {
            "requested_at": demo_time.isoformat(),
            "as_of_version": {"open": correct.open, "effective_at": _iso(correct.version_time)},
            "latest_version_if_leaked": {"open": latest.open, "effective_at": _iso(latest.version_time)},
            "difference": "按最新版本会把 f 误判为闭店并排除；as-of 版本仍营业。",
        }
    )
    return {
        "split_membership_overlaps": report["overlaps"],
        "overlap_free": not report["overlaps"],
        "same_request_in_multiple_splits": any(
            len([1 for rows in splits.as_dict().values() if any(i.request_id == rid for i in rows)]) > 1
            for rid in {i.request_id for rows in splits.as_dict().values() for i in rows}
        ),
        "time_ordering_ok": ordered,
        "bounds": {
            "train": [_iso(tmin), _iso(tmax)],
            "validation": [_iso(vmin), _iso(vmax)],
            "test": [_iso(smin), _iso(smax)],
        },
        "attribute_version_violations": violations,
        "future_version_counterfactual": counterfactual,
        "pass": not report["overlaps"] and ordered and not violations,
        "note": "检查覆盖每条请求的每个候选属性；切分以请求时间为唯一依据。",
    }


def _iso(value):
    return None if value is None else value.isoformat()


def run_policy_scenarios(impressions, test_impressions, cfg, uniform_verified,
                         ranker_decisions) -> dict:
    repeats, seed = cfg["bootstrap_repeats"], cfg["seed"]

    # 场景 1：候选集均匀随机目标。在均匀记录核验下，候选内每个动作的 μ 已知且为正。
    def uniform_target(imp):
        return {item: 1.0 / len(imp.candidates) for item in imp.candidates}

    uniform = ope.evaluate_policy(
        impressions, "reward_click", uniform_target, repeats, seed,
        _rng_factory(seed), uniform_verified=uniform_verified, clip=5.0,
    )

    # 场景 2：不可支持目标——推荐一批日志从未曝光的新项目。
    # 这些动作不在任何候选集内，记录概率为 0（由均匀核验推得），目标价值不可识别。
    def new_item_target(imp):
        return {"new-merchant-never-logged": 1.0}

    unsupported = ope.evaluate_policy(
        impressions, "reward_click", new_item_target, repeats, seed,
        _rng_factory(seed), uniform_verified=uniform_verified, clip=1.0,
    )
    unsupported["scenario"] = (
        "目标只推荐从未出现在日志候选集的新项目。零支持质量使其价值不可识别；"
        "不得把 0 值读成“新店没有价值”。"
    )

    # 场景 3：部署展示列表 + 均匀抽取（测试集）。动作概率含重排与最终抽取。
    display = ope.evaluate_policy(
        test_impressions, "reward_click", _display_target(ranker_decisions), repeats, seed,
        _rng_factory(seed), uniform_verified=uniform_verified, clip=2.0,
    )

    # 场景 4：随机列表策略的边际动作概率（含重排 + 最终抽取），同样按展示目标评估。
    randomized = ope.evaluate_policy(
        test_impressions, "reward_click", _randomized_target(ranker_decisions), repeats, seed,
        _rng_factory(seed), uniform_verified=uniform_verified, clip=2.0,
    )
    return {
        "uniform_random_target": uniform,
        "unsupported_new_item_target": unsupported,
        "deployment_display_target": display,
        "randomized_list_target": randomized,
    }


def run_item_version_demo(catalog: ItemCatalog, splits: Splits) -> dict:
    def at(value):
        return catalog.as_of("f", parse_time(value, "demo"))

    per_request = []
    for name, rows in splits.as_dict().items():
        for imp in rows:
            if "f" in imp.candidates:
                availability = resolve_availability(imp, catalog)
                per_request.append({
                    "split": name,
                    "request_id": imp.request_id,
                    "requested_at": imp.requested_at.isoformat(),
                    "f_open": "f" not in availability.excluded,
                    "f_exclusion": availability.excluded.get("f"),
                })
    return {
        "before_2026-10-20": _snapshot(at("2026-10-19T00:00:00+08:00")),
        "from_2026-10-20": _snapshot(at("2026-10-20T00:00:00+08:00")),
        "after_known_at": _snapshot(at("2026-10-25T00:00:00+08:00")),
        "per_request": per_request,
        "note": "闭店版本 known_at=2026-10-19、effective_at=2026-10-20；"
                "只影响 2026-10-20 起的请求，此前请求仍使用营业版本。",
    }


def _snapshot(snapshot):
    return {
        "item": snapshot.item_key,
        "category": snapshot.category,
        "open": snapshot.open,
        "effective_at": _iso(snapshot.version_time),
        "known_at": _iso(snapshot.known_at),
    }


def run_cold_start_demo(catalog: ItemCatalog, baseline, ranker, train_impressions) -> dict:
    """训练集从未出现的新项目：基线与排序器都只能给出中性先验，不继承未来点击。

    演示小时取自训练集实际观测到的时段，避免用测试/未来时刻构造特征。
    """
    known_hours = sorted({imp.requested_at.hour for imp in train_impressions})
    hour = known_hours[0]
    demo_time = parse_time("2026-10-21T10:00:00+08:00", "demo")
    synthetic = ItemVersion(
        item_key="g", category="dining",
        effective_at=parse_time("2026-10-21T00:00:00+08:00", "demo"),
        known_at=parse_time("2026-10-21T00:00:00+08:00", "demo"),
        open=True,
    )
    extended = ItemCatalog(list(catalog.versions()) + [synthetic])
    snapshot = extended.as_of("g", demo_time)
    baseline_score = baseline.score("g")
    template = cold_start_template(ranker, "dining", known_hours)
    ranker_score = template["score"]
    # 反事实：若违规用未来热门度回填（假设新项目 100 次曝光、90 次点击）。
    leaked_feature_score = (0.9 * 100 + baseline.prior_mean * baseline.pseudo_views) / (
        100 + baseline.pseudo_views
    )
    return {
        "synthetic_item": "g（仅用于演示，不进入任何评估集合）",
        "base_rate_rule": baseline.explain("g"),
        "baseline_score": baseline_score,
        "ranker_cold_start_score": ranker_score,
        "ranker_features_used": {
            "item_click_rate_train": 0.5,
            "item_redeem_rate_train": 0.5,
            "log1p_train_views": 0.0,
            "category": template["given_category"],
            "hour_bucket_from_training_hours": [h for h in known_hours],
            "category_known_in_training": template["category_known_in_training"],
            "reason": "训练集无该项目，率与曝光特征取中性值；仅业态与训练已见的时段可贡献分数",
        },
        "if_future_popularity_were_backfilled": leaked_feature_score,
        "difference_if_backfilled": leaked_feature_score - baseline_score,
        "note": "回填未来热门度会把冷启动项目分数抬高，属于时间穿越；流水线拒绝该做法。",
    }


def run_fill_shortfall_demo(catalog: ItemCatalog, cfg) -> dict:
    """合成演示：3 个同业态候选、上限 2、请求长度 3，必然填不满。"""
    demo = Impression(
        request_id="demo-fill-shortfall",
        requested_at=parse_time("2026-10-19T10:00:00+08:00", "demo"),
        candidates=("a", "b", "c"),
        selected_item="a",
        logging_propensity=1.0 / 3.0,
        reward_click=0,
        reward_redeem=0,
    )
    decision = build_list_policy(
        demo, catalog, {"a": 0.9, "b": 0.8, "c": 0.7},
        list_size=cfg["list_size"], max_per_category=cfg["max_per_category"],
    )
    return {
        "request": {
            "request_id": demo.request_id,
            "candidates": list(demo.candidates),
            "categories": {"a": "dining", "b": "dining", "c": "dining"},
            "requested_list_size": cfg["list_size"],
            "max_per_category": cfg["max_per_category"],
            "note": "合成演示，不属于 fixtures 评估集",
        },
        "display_list": decision.display,
        "fill": decision.fill,
        "action_probs": decision.deterministic_draw_probs,
    }


def _weighted_log_loss(score_fn, impressions, catalog) -> float:
    total, weight_sum = 0.0, 0.0
    for imp in impressions:
        score = score_fn(imp)[imp.selected_item]
        p = min(max(score, 1e-6), 1 - 1e-6)
        target = float(imp.reward_click)
        weight = 1.0 / imp.logging_propensity
        total -= weight * (target * math.log(p) + (1 - target) * math.log(1 - p))
        weight_sum += weight
    return total / weight_sum if weight_sum else float("nan")


def _compare_rankers(test, baseline_scores, learned_scores, catalog) -> dict:
    per_request = []
    baseline_mrr = ranker_mrr = 0.0
    baseline_first = ranker_first = 0
    for imp in test:
        bs = baseline_scores(imp)
        rs = learned_scores(imp)
        base_ranked = sorted(imp.candidates, key=lambda item: (-bs[item], item))
        learned_ranked = sorted(imp.candidates, key=lambda item: (-rs[item], item))
        base_pos = base_ranked.index(imp.selected_item) + 1
        learned_pos = learned_ranked.index(imp.selected_item) + 1
        baseline_mrr += 1.0 / base_pos
        ranker_mrr += 1.0 / learned_pos
        baseline_first += base_pos == 1
        ranker_first += learned_pos == 1
        per_request.append(
            {
                "request_id": imp.request_id,
                "logged_item": imp.selected_item,
                "baseline_score": bs[imp.selected_item],
                "ranker_score": rs[imp.selected_item],
                "baseline_rank_of_logged_item": base_pos,
                "ranker_rank_of_logged_item": learned_pos,
                "baseline_ranking": base_ranked,
                "ranker_ranking": learned_ranked,
            }
        )
    n = len(test) or 1
    return {
        "predictive_weighted_log_loss": {
            "baseline_popularity": _weighted_log_loss(baseline_scores, test, catalog),
            "learned_ranker": _weighted_log_loss(learned_scores, test, catalog),
            "note": "1/propensity 加权对数损失，越低越好；衡量预测质量。",
        },
        "ranking": {
            "baseline_popularity": {
                "mean_reciprocal_rank_of_logged_action": baseline_mrr / n,
                "logged_action_ranked_first": baseline_first / n,
            },
            "learned_ranker": {
                "mean_reciprocal_rank_of_logged_action": ranker_mrr / n,
                "logged_action_ranked_first": ranker_first / n,
            },
            "requests": len(test),
            "note": "排序指标只衡量对日志选中项目的预测排序，不等于策略价值。",
        },
        "per_request": per_request,
        "interpretation": (
            "预测排序更好的排序器，其目标列表在日志中的回报估计也可能更高；"
            "但两者是不同问题：日志选中项目本身受记录策略影响，排序指标高不代表线上因果收益。"
        ),
    }


def _pre_rerank_counterexample(impressions, decisions, score_fn, catalog, cfg) -> dict:
    """显式对比：用重排前的归一化分数冒充动作概率，会高估被业态上限挤掉的项目。"""
    examples = []
    for imp in impressions:
        decision = decisions[imp.request_id]
        if not decision.display:
            continue
        scores = score_fn(imp)
        if not scores:
            continue
        peak = max(scores.values())
        exps = {item: math.exp((scores[item] - peak) / 0.10) for item in scores}
        total = sum(exps.values())
        pre = {item: value / total for item, value in exps.items()}
        mismatch = any(
            abs(pre.get(item, 0.0) - decision.deterministic_draw_probs.get(item, 0.0)) > 1e-9
            for item in imp.candidates
        )
        if mismatch:
            examples.append(
                {
                    "request_id": imp.request_id,
                    "display_list": decision.display,
                    "pre_rerank_normalized_scores": pre,
                    "actual_draw_probs": decision.deterministic_draw_probs,
                    "difference": {
                        item: pre.get(item, 0.0) - decision.deterministic_draw_probs.get(item, 0.0)
                        for item in imp.candidates
                    },
                }
            )
    return {
        "examples": examples,
        "note": "目标动作概率必须是“列表概率 × 1/列表长度”；重排前分数不含约束，"
                "会造成动作概率与真实抽取机制不一致。",
    }


def _computation_design(decisions, cfg, mc_demo) -> dict:
    methods = {}
    total = 0.0
    for decision in decisions.values():
        method = decision.estimate.get("method", "unknown")
        entry = methods.setdefault(
            method,
            {"requests": 0, "enumerated_lists": 0, "sampled_lists": 0,
             "estimated_ops": 0, "max_mc_stderr": 0.0},
        )
        entry["requests"] += 1
        entry["enumerated_lists"] += decision.estimate.get("enumerated_lists", 0)
        entry["sampled_lists"] += decision.estimate.get("sampled_lists", 0)
        entry["estimated_ops"] += decision.estimate.get("estimated_ops", 0)
        entry["max_mc_stderr"] = max(
            entry["max_mc_stderr"],
            decision.estimate.get("monte_carlo_standard_error_max", 0.0),
        )
        total += decision.timing_seconds
    return {
        "paths": {
            "deployment_display": "无需枚举：展示列表按分数与约束确定，π(a)=1/|列表|，精确",
            "randomized_list_policy": "合法列表价值 softmax，再均匀抽取；π(a)=Σ_L w(L)/|L|",
        },
        "exact_enumeration_budget_lists": cfg.get("max_exact_lists", 50000),
        "methods": methods,
        "wall_clock_seconds": round(total, 6),
        "monte_carlo_demonstration": mc_demo,
        "budget_statement": (
            "枚举成本 ≈ 合法列表数 × 列表长度；抽样成本 ≈ 样本数 / 接受率 × 列表长度。"
            "合法列表数由按业态分组的 DP 先算出，超过预算才转为拒绝采样。"
        ),
    }


def _monte_carlo_demo(test, catalog, score_fn, cfg, seed) -> dict:
    """强制走抽样路径，展示动作概率的蒙特卡洛误差。"""
    examples = []
    for imp in test[:2]:
        decision = build_list_policy(
            imp, catalog, score_fn(imp),
            list_size=cfg["list_size"], max_per_category=cfg["max_per_category"],
            max_exact_lists=1, mc_samples=2000, seed=seed,
        )
        examples.append({
            "request_id": imp.request_id,
            "method": decision.estimate["method"],
            "legal_list_count": decision.estimate["legal_list_count"],
            "sampled_lists": decision.estimate.get("sampled_lists"),
            "acceptance_rate": decision.estimate.get("acceptance_rate"),
            "action_probs": decision.randomized_action_probs,
            "monte_carlo_standard_error_by_action": decision.estimate.get(
                "monte_carlo_standard_error_by_action"
            ),
            "action_prob_error": decision.estimate.get("action_prob_error"),
        })
    return {
        "note": "把精确枚举预算压到 1 后强制抽样；标准误约为 1/sqrt(样本数) 量级。",
        "examples": examples,
    }


@dataclass
class PipelineResult:
    payload: dict
    text_summary: str


def _json_default(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(type(value).__name__)


def run_pipeline(fixture_dir: Path, output_dir: Path | None = None,
                 seed_override: int | None = None,
                 max_exact_lists: int = 50000) -> PipelineResult:
    raw_impressions, raw_items, raw_policies = load_fixture_dir(Path(fixture_dir))
    cfg = read_policies(raw_policies)
    cfg["max_exact_lists"] = max_exact_lists
    if seed_override is not None:
        cfg["seed"] = int(seed_override)
    catalog = load_catalog(raw_items)
    impressions, duplicates = deduplicate(raw_impressions)
    check_item_references(impressions, catalog)
    splits = make_splits(impressions, cfg["train_end"], cfg["validation_end"], cfg["test_end"])
    if not splits.train:
        raise InputContractError("训练集为空：检查 train_end")

    uniform_check = ope.verify_logging_policy(impressions)
    uniform_verified = uniform_check["verified_uniform_over_candidates"]

    baseline = train_popularity_baseline(splits.train)
    ranker, tuning = tune_ranker(splits.train, splits.validation, catalog)

    def baseline_scores(imp):
        return baseline_scores_for_request(baseline, imp, catalog)

    def learned_scores(imp):
        return ranker_scores(ranker, imp, catalog)

    seed = cfg["seed"]
    baseline_decisions = _decisions(splits.test, catalog, baseline_scores, cfg, seed, max_exact_lists)
    ranker_decisions = _decisions(splits.test, catalog, learned_scores, cfg, seed, max_exact_lists)

    evaluation = {}
    for name, decisions in (("baseline_popularity", baseline_decisions),
                            ("learned_ranker", ranker_decisions)):
        entry = ope.evaluate_policy(
            splits.test, "reward_click", _display_target(decisions),
            cfg["bootstrap_repeats"], seed, _rng_factory(seed),
            uniform_verified=uniform_verified, clip=2.0,
        )
        entry["target_definition"] = (
            "部署展示策略：π(a)=1/|最终列表|（a 在列表中）；已含重排约束与最终均匀抽取"
        )
        entry["reading_guidance"] = (
            "测试集只有 6 个请求，且每个请求只记录 1 个动作；"
            "IPS/SNIPS 的点估计与区间都很不稳定，最大值 1.0 不代表线上最佳。"
            "该数值只说明：在给定记录策略与目标策略下，对日志选中动作的加权回报估计。"
        )
        entry["randomized_list_policy"] = ope.evaluate_policy(
            splits.test, "reward_click", _randomized_target(decisions),
            cfg["bootstrap_repeats"], seed, _rng_factory(seed),
            uniform_verified=uniform_verified, clip=2.0,
        )
        evaluation[name] = entry

    payload = {
        "inputs": {
            "fixture_dir": str(fixture_dir),
            "raw_rows": len(raw_impressions),
            "unique_requests": len(impressions),
            "removed_duplicate_rows": duplicates,
        },
        "splits": {name: {"rows": len(rows), "requests": [i.request_id for i in rows]}
                   for name, rows in splits.as_dict().items()},
        "logging_policy_check": uniform_check,
        "hand_check": run_hand_check(impressions, cfg, uniform_verified),
        "time_leakage_audit": run_time_leakage_audit(splits, catalog),
        "policy_scenarios": run_policy_scenarios(
            impressions, splits.test, cfg, uniform_verified, ranker_decisions
        ),
        "item_version_demo": run_item_version_demo(catalog, splits),
        "cold_start_demo": run_cold_start_demo(catalog, baseline, ranker, splits.train),
        "fill_shortfall_demo": run_fill_shortfall_demo(catalog, cfg),
        "baseline_model": {
            "prior_mean": baseline.prior_mean,
            "pseudo_views": baseline.pseudo_views,
            "train_stats": {
                item: baseline.explain(item) for item in sorted(baseline.logged_selections)
            },
            "cold_start_rule": "训练集未出现项目取先验 1/2；不继承未来点击",
        },
        "learned_ranker": {
            "feature_names": ranker.names,
            "weights": dict(zip(ranker.names, ranker.weights)),
            "training": ranker.training,
            "tuning": tuning,
        },
        "test_evaluation": evaluation,
        "ranker_comparison": _compare_rankers(splits.test, baseline_scores, learned_scores, catalog),
        "pre_rerank_counterexample": _pre_rerank_counterexample(
            splits.test, ranker_decisions, learned_scores, catalog, cfg
        ),
        "test_list_examples": [
            decision_to_dict(ranker_decisions[imp.request_id], learned_scores(imp))
            for imp in splits.test
        ],
        "computation_design": _computation_design(
            ranker_decisions, cfg,
            _monte_carlo_demo(splits.test, catalog, learned_scores, cfg, seed),
        ),
        "claims": [
            "日志是观察性的；所有 OPE 数值都依赖记录策略与目标概率的给定假设，"
            "不是线上因果收益的证明。",
            "目标动作概率同时包含重排列表概率与最终均匀抽取，不能使用重排前分数。",
            "支持损失存在时只报告已识别部分，拒绝称完整策略收益。",
            "裁剪后的 IPS 是有偏估计量，与未裁剪结果分列，不标注为无偏。",
            "冷启动项目使用中性先验，不使用未来点击或热门度回填。",
        ],
    }
    summary = render_summary(payload)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "report.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    return PipelineResult(payload=payload, text_summary=summary)


def _fmt(value, digits=6):
    return "none" if value is None else f"{value:.{digits}f}"


def render_summary(payload: dict) -> str:
    lines = ["# 跨业态推荐：离线评估摘要", "",
             "所有数值来自合成资料，是离线估计，不是线上因果收益证明。", "",
             "## 数据与切分"]
    inputs = payload["inputs"]
    lines.append(f"- 原始曝光行 {inputs['raw_rows']}，去重后请求 {inputs['unique_requests']}，"
                 f"移除重复行 {inputs['removed_duplicate_rows']}")
    for name, info in payload["splits"].items():
        lines.append(f"- {name}: {info['rows']} 个请求")
    lines += ["", "## 记录策略核验与手算校验"]
    check = payload["logging_policy_check"]
    lines.append(f"- 记录策略与候选集均匀一致: {check['verified_uniform_over_candidates']}"
                 f"（偏离 {check['deviation_count']} 条）")
    a = payload["hand_check"]["assertions"]
    lines.append(f"- IPS={_fmt(a['ips_actual'])}（期望 0.25，通过={a['ips_ok']}）；"
                 f"SNIPS={_fmt(a['snips_actual'])}（期望 1/3，通过={a['snips_ok']}）")
    lines.append(f"- {payload['hand_check']['interpretation']}")
    lines += ["", "## 时间穿越检查"]
    audit = payload["time_leakage_audit"]
    lines.append(f"- 切分重叠: {audit['split_membership_overlaps'] or '无'}；"
                 f"同请求跨集合: {audit['same_request_in_multiple_splits']}")
    lines.append(f"- 属性版本违规: {len(audit['attribute_version_violations'])} 条；"
                 f"时间顺序: {audit['time_ordering_ok']}；结论: {'通过' if audit['pass'] else '未通过'}")
    demo = audit["future_version_counterfactual"][0]
    lines.append(f"- 反事实: {demo['difference']}")
    lines += ["", "## 不可支持策略与支持损失"]
    unsupported = payload["policy_scenarios"]["unsupported_new_item_target"]
    support = unsupported["support"]
    lines.append(f"- 状态: {support['status']}；完整策略价值可识别={support['full_policy_value_identifiable']}")
    lines.append(f"- 可证明零支持请求: {support['proven_zero_support_request_count']} 个，"
                 f"零支持质量合计 {support['proven_zero_support_mass_total']:.3f}")
    lines.append(f"- {support['statement']}")
    full = payload["policy_scenarios"]["uniform_random_target"]["support"]
    lines.append(f"- 对照（候选集均匀随机目标）: {full['status']}，"
                 f"完整价值可识别={full['full_policy_value_identifiable']}")
    lines += ["", "## 测试集离线指标（目标概率含重排与最终抽取）"]
    for name, entry in payload["test_evaluation"].items():
        plain = entry["unclipped"]
        boot = plain["bootstrap"]["ips"]
        clipped = entry.get("clipped", {})
        lines.append(f"### {name}")
        lines.append(f"- 部署展示目标：IPS={_fmt(plain['ips'])}，SNIPS={_fmt(plain['snips'])}，"
                     f"ESS={_fmt(plain['ess'], 2)}（{_fmt(plain['ess_ratio'] * 100, 1)}%），"
                     f"覆盖={_fmt(plain['coverage'] * 100, 1)}%")
        lines.append(f"- IPS 95% 按请求 bootstrap 区间 [{_fmt(boot['low'])}, {_fmt(boot['high'])}]")
        if clipped:
            cb = clipped["bootstrap"]["ips"]
            lines.append(f"- 裁剪@{clipped['clip_threshold']}：IPS={_fmt(clipped['ips'])}"
                         f"（有偏；区间 [{_fmt(cb['low'])}, {_fmt(cb['high'])}]）")
        four = entry["randomized_list_policy"]["unclipped"]
        lines.append(f"- 随机列表策略：IPS={_fmt(four['ips'])}，SNIPS={_fmt(four['snips'])}，"
                     f"ESS={_fmt(four['ess'], 2)}")
        lines.append(f"- 支持状态: {entry['support']['status']}；"
                     f"完整价值可识别={entry['support']['full_policy_value_identifiable']}")
        lines.append(f"- 判读: {entry['reading_guidance']}")
    lines += ["", "## 排序器对比（预测质量 vs 策略价值）"]
    comp = payload["ranker_comparison"]
    b = comp["predictive_weighted_log_loss"]
    lines.append(f"- 加权对数损失：基线 {_fmt(b['baseline_popularity'])}，"
                 f"学习排序器 {_fmt(b['learned_ranker'])}")
    m = comp["ranking"]
    lines.append(f"- MRR：基线 {_fmt(m['baseline_popularity']['mean_reciprocal_rank_of_logged_action'], 3)}"
                 f"，排序器 {_fmt(m['learned_ranker']['mean_reciprocal_rank_of_logged_action'], 3)}；"
                 f"排第一比例 基线 {_fmt(m['baseline_popularity']['logged_action_ranked_first'] * 100, 1)}%"
                 f"，排序器 {_fmt(m['learned_ranker']['logged_action_ranked_first'] * 100, 1)}%")
    lines.append(f"- {comp['interpretation']}")
    lines += ["", "## 约束重排与动作概率"]
    f = payload["fill_shortfall_demo"]
    lines.append(f"- 无法填满演示：请求长度 {f['request']['requested_list_size']}，"
                 f"可用 {f['fill']['available_count']} 个，最终 {f['fill']['final_size']} 个")
    for reason in f["fill"]["reasons"]:
        lines.append(f"  - 原因：{reason}")
    pre = payload["pre_rerank_counterexample"]
    lines.append(f"- 重排前概率反例：{len(pre['examples'])} 个请求中，重排前分数与真实抽取概率不同；"
                 "不能冒充动作概率")
    comp_design = payload["computation_design"]
    for method, info in comp_design["methods"].items():
        lines.append(f"- {method}: 请求 {info['requests']}，枚举 {info['enumerated_lists']}，"
                     f"抽样 {info['sampled_lists']}，最大蒙特卡洛标准误 {info['max_mc_stderr']:.3g}")
    lines.append(f"- {comp_design['budget_statement']}")
    lines += ["", "## 属性版本切换与冷启动"]
    ver = payload["item_version_demo"]
    lines.append(f"- f 在 2026-10-20 前 open={ver['before_2026-10-20']['open']}，"
                 f"之后 open={ver['from_2026-10-20']['open']}")
    cold = payload["cold_start_demo"]
    lines.append(f"- 冷启动项目基线分数={_fmt(cold['baseline_score'])}，"
                 f"若违规回填未来热门度会变成 {_fmt(cold['if_future_popularity_were_backfilled'])}")
    lines += ["", "## 声明"]
    for claim in payload["claims"]:
        lines.append(f"- {claim}")
    return "\n".join(lines) + "\n"
