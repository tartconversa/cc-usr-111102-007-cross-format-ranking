"""离策略评估：IPS、SNIPS、有效样本量、覆盖、按请求 bootstrap，以及支持性判定。

三层区分（这是本模块的核心）：
1. 记录策略支持：日志每行只给被选动作的 propensity。若逐行核验到
   propensity == 1/|候选集|，则可判定记录策略是“候选集上均匀随机”，
   从而所有候选动作的 μ(a) 都已知且为正；否则只有被选动作的 μ 已知。
2. 目标支持：目标策略 π 在哪些动作上有正概率。若 π 的正质量落在
   μ(a)=0 的动作上（动作不在候选集，或已核验确定性记录），这部分价值不可识别。
   若落在 μ 未知的动作上，也无法核验，按“不可识别”处理。
3. 估计量：IPS/SNIPS 只用被记录动作的 π/μ 权重，因此即使目标支持未完全核验，
   数值也只等于“已识别部分”；报告必须据此标注，不能称完整策略收益。

权重裁剪是另一个估计量，单独列出并标注有偏。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .data import Impression

UNIFORM_TOLERANCE = 1e-9
# 目标在“被记录动作之外、但仍在候选集之内”的质量：分布未知，但动作空间已知。
CANDIDATE_REMAINDER = "__candidate_remainder__"


def verify_logging_policy(impressions: list[Impression]) -> dict:
    """核验记录策略是否为候选集上的均匀随机。

    只做可观察的检查：每行 propensity 是否等于 1/|候选集|。
    结论是“与均匀一致”，而不是“证明了线上记录策略就是这样”；
    若存在偏差，则退化为只有被选动作概率已知。
    """
    deviations = []
    for imp in impressions:
        expected = 1.0 / len(imp.candidates)
        if abs(imp.logging_propensity - expected) > UNIFORM_TOLERANCE:
            deviations.append(
                {
                    "request_id": imp.request_id,
                    "candidate_count": len(imp.candidates),
                    "recorded_propensity": imp.logging_propensity,
                    "uniform_would_be": expected,
                }
            )
    return {
        "verified_uniform_over_candidates": not deviations,
        "deviation_count": len(deviations),
        "deviations": deviations,
        "statement": (
            "所有记录的 propensity 等于 1/|候选集|，与候选集均匀随机一致；"
            "在此核验下，候选集内每个动作的 μ 均为 1/|候选集|。"
            if not deviations
            else "存在偏离均匀的请求：只能核验被选动作的记录概率，其它动作的支持未知。"
        ),
    }


def known_logging_probs(impression: Impression, uniform_verified: bool) -> dict[str, float]:
    if uniform_verified:
        share = 1.0 / len(impression.candidates)
        return {item: share for item in impression.candidates}
    return {impression.selected_item: impression.logging_propensity}


@dataclass(frozen=True)
class PerRequest:
    request_id: str
    reward: float
    target_prob: float
    logging_prob: float
    weight: float
    target_support: dict[str, float]
    known_logging: dict[str, float]
    unobserved_mass: float = 0.0
    candidate_actions: tuple[str, ...] = ()


def per_request_rows(impressions, reward_name, target_probs_for,
                     uniform_verified: bool) -> list[PerRequest]:
    rows = []
    for impression in impressions:
        reward = float(getattr(impression, reward_name))
        support = dict(target_probs_for(impression))
        remainder = support.pop(CANDIDATE_REMAINDER, 0.0)
        if remainder < 0:
            raise ValueError(f"{impression.request_id}: 候选集内其余目标质量不能为负")
        total = sum(support.values()) + remainder
        if total > 1.0 + 1e-9 or any(p < 0 for p in support.values()):
            raise ValueError(f"{impression.request_id}: 目标动作概率非法：{support}")
        logging_probs = known_logging_probs(impression, uniform_verified)
        target_prob = support.get(impression.selected_item, 0.0)
        mu = logging_probs.get(impression.selected_item)
        weight = target_prob / mu if mu else float("inf")
        rows.append(
            PerRequest(
                request_id=impression.request_id,
                reward=reward,
                target_prob=target_prob,
                logging_prob=impression.logging_propensity,
                weight=weight,
                target_support=support,
                known_logging=logging_probs,
                unobserved_mass=remainder,
                candidate_actions=impression.candidates,
            )
        )
    return rows


def support_report(rows: list[PerRequest], uniform_verified: bool) -> dict:
    proven_zero = []
    unverifiable = []
    for row in rows:
        zero_mass = 0.0
        zero_actions = []
        unknown_mass = 0.0
        unknown_actions = []
        for action, prob in row.target_support.items():
            if prob <= 0:
                continue
            mu = row.known_logging.get(action)
            if mu is None:
                if uniform_verified and action not in row.candidate_actions:
                    # 记录策略在候选集上均匀，候选集外动作的记录概率确定为 0。
                    zero_mass += prob
                    zero_actions.append(action)
                else:
                    unknown_mass += prob
                    unknown_actions.append(action)
            elif mu <= 0:
                zero_mass += prob
                zero_actions.append(action)
        if zero_mass > 0:
            proven_zero.append(
                {
                    "request_id": row.request_id,
                    "zero_support_mass": zero_mass,
                    "zero_support_actions": zero_actions,
                    "target_prob_on_logged_action": row.target_prob,
                }
            )
        if unknown_mass > 0:
            unverifiable.append(
                {
                    "request_id": row.request_id,
                    "unknown_support_mass": unknown_mass,
                    "unknown_support_actions": unknown_actions,
                }
            )
        # 候选集内其余质量：均匀核验下所有候选动作的 μ 已知且为正，视为有支持；
        # 否则记录概率不可核验。
        if row.unobserved_mass > 0 and not uniform_verified:
            unverifiable.append(
                {
                    "request_id": row.request_id,
                    "unknown_support_mass": row.unobserved_mass,
                    "unknown_support_actions": [],
                    "candidate_remainder_mass": row.unobserved_mass,
                }
            )
    if proven_zero:
        status = "proven_zero_support"
        full = False
        statement = (
            "目标在日志零支持动作上有正概率：这部分价值不可识别。"
            "表中 IPS/SNIPS 只是已识别部分，禁止当作完整策略收益。"
        )
    elif unverifiable:
        status = "support_unverified"
        full = False
        statement = (
            "目标质量落在记录概率未知的动作上，无法核验其支持；"
            "表中数值只覆盖被记录动作，不能称完整策略收益。"
        )
    else:
        status = "verified_full_support"
        full = True
        statement = (
            "目标支持集内每个动作的记录概率均已核验为正，"
            "在此核验下 IPS 是完整目标价值的无偏估计（仍仅限离线、无模型错设之外的假设）。"
        )
    return {
        "logging_policy_uniform_verified": uniform_verified,
        "status": status,
        "proven_zero_support_requests": proven_zero,
        "proven_zero_support_request_count": len(proven_zero),
        "proven_zero_support_mass_total": sum(v["zero_support_mass"] for v in proven_zero),
        "unverified_support_requests": unverifiable,
        "unverified_support_request_count": len(unverifiable),
        "full_policy_value_identifiable": full,
        "value_reporting": "full" if full else "deny_full_policy_value",
        "statement": statement,
    }


def _estimate(rows: list[PerRequest]) -> dict:
    n = len(rows)
    if n == 0:
        return {"requests": 0, "ips": None, "snips": None, "ess": None,
                "coverage": None, "zero_propensity_rows": 0, "max_weight": None}
    finite = all(math.isfinite(row.weight) for row in rows)
    rewards = [row.reward for row in rows]
    weights = [row.weight for row in rows]
    if finite:
        ips = sum(w * r for w, r in zip(weights, rewards)) / n
        weight_sum = sum(weights)
        snips = (
            sum(w * r for w, r in zip(weights, rewards)) / weight_sum
            if weight_sum else None
        )
        ess = (weight_sum ** 2) / sum(w * w for w in weights) if weight_sum else 0.0
    else:
        ips = snips = ess = None
        weight_sum = None
    return {
        "requests": n,
        "ips": ips,
        "snips": snips,
        "ess": ess,
        "ess_ratio": (ess / n) if ess is not None else None,
        "coverage": sum(1 for row in rows if row.target_prob > 0) / n,
        "zero_propensity_rows": sum(1 for row in rows if row.logging_prob <= 0.0),
        "infinite_weight_rows": sum(1 for row in rows if not math.isfinite(row.weight)),
        "max_weight": max(weights) if weights and finite else None,
        "weight_sum": weight_sum,
    }


def _percentile_interval(values: list[float]) -> dict:
    cleaned = sorted(v for v in values if v is not None and math.isfinite(v))
    if not cleaned:
        return {"low": None, "high": None}
    def quantile(q):
        position = q * (len(cleaned) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return cleaned[lower]
        return cleaned[lower] * (upper - position) + cleaned[upper] * (position - lower)
    return {"low": quantile(0.025), "high": quantile(0.975)}


def _bootstrap(rows: list[PerRequest], repeats: int, seed: int, rng_factory) -> dict:
    ids = sorted({row.request_id for row in rows})
    if not ids:
        return {"repeats": 0}
    by_id: dict[str, list[PerRequest]] = {}
    for row in rows:
        by_id.setdefault(row.request_id, []).append(row)
    rng = rng_factory(seed)
    ips_values, snips_values = [], []
    for _ in range(repeats):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        pooled = [row for request_id in sample for row in by_id[request_id]]
        value = _estimate(pooled)
        ips_values.append(value["ips"])
        snips_values.append(value["snips"])
    return {"repeats": repeats, "resampled_requests": len(ids),
            "ips": _percentile_interval(ips_values),
            "snips": _percentile_interval(snips_values)}


def evaluate_policy(impressions, reward_name, target_probs_for, repeats, seed,
                    rng_factory, uniform_verified: bool, clip: float | None = None) -> dict:
    rows = per_request_rows(impressions, reward_name, target_probs_for, uniform_verified)
    support = support_report(rows, uniform_verified)
    plain = _estimate(rows)
    plain["bootstrap"] = _bootstrap(rows, repeats, seed, rng_factory)
    result = {
        "reward": reward_name,
        "unclipped": {
            **plain,
            "bias": (
                "在支持核验通过时无偏（针对给定记录策略与目标策略）；"
                "支持不足时只读作已识别下界"
            ),
            "full_policy_value_identifiable": support["full_policy_value_identifiable"],
        },
        "support": support,
        "hand_verifiable": hand_check(rows),
    }
    if clip is not None:
        clipped_rows = [
            PerRequest(row.request_id, row.reward, row.target_prob, row.logging_prob,
                       min(row.weight, clip) if math.isfinite(row.weight) else clip,
                       row.target_support, row.known_logging, row.unobserved_mass,
                       row.candidate_actions)
            for row in rows
        ]
        clipped = _estimate(clipped_rows)
        clipped["bootstrap"] = _bootstrap(clipped_rows, repeats, seed, rng_factory)
        result["clipped"] = {
            **clipped,
            "clip_threshold": clip,
            "bias": "裁剪引入偏差（有偏估计量）；数值不能当成无偏策略收益，区间也只描述裁剪后估计量",
        }
    return result


def hand_check(rows: list[PerRequest]) -> dict:
    if not rows:
        return {"rows": [], "ips": None, "snips": None}
    n = len(rows)
    weight_sum = sum(row.weight for row in rows)
    return {
        "rows": [
            {
                "request_id": row.request_id,
                "reward": row.reward,
                "target_prob": row.target_prob,
                "logging_prob": row.logging_prob,
                "weight": row.weight,
                "contribution_to_ips": row.weight * row.reward / n,
            }
            for row in rows
        ],
        "ips": sum(row.weight * row.reward for row in rows) / n,
        "snips": (
            sum(row.weight * row.reward for row in rows) / weight_sum
            if weight_sum else None
        ),
        "weight_sum": weight_sum,
        "ess": (weight_sum ** 2) / sum(row.weight * row.weight for row in rows)
        if weight_sum else 0.0,
    }
