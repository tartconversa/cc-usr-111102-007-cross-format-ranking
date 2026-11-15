"""约束重排 -> 可计算动作概率。

关键设计：把“最终列表”上的约束转成项目级动作概率，分两条可核验的路径。

路径 A（部署展示，主要评估对象）：
  display_list = 按分数降序、至多 max_per_category 个同业态、排除闭店/无可见版本的合法列表。
  目标策略 = 从该列表均匀抽取一项，故 π(a) = 1/|display_list|（a 在列表中），否则 0。
  这条路径无需枚举，动作概率精确、可手算，不产生抽样误差。

路径 B（随机列表策略，用于探索/学习）：
  在“全部合法列表”上按列表价值的 softmax 加权，再均匀抽取一项：
      π(a) = Σ_{L: a∈L} w(L) / |L|,   w(L) = softmax(mean_score(L)/τ)
  合法列表数在预算内时精确枚举（误差 0）；超出预算时用固定种子拒绝采样，
  给出每个动作的蒙特卡洛标准误并显式标注。

两条路径的动作概率都包含“重排 + 最终抽取”两层，不用重排前分数冒充。
"""

from __future__ import annotations

import hashlib
import itertools
import math
import random
import time
from dataclasses import dataclass, field

from .data import ItemCatalog, Impression, AsOfItem
from .errors import NoVersionError

DEFAULT_EXACT_CAP = 50000
DEFAULT_MC_SAMPLES = 4000


def stable_seed(seed: int, text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (seed * 1000003 + int(digest[:16], 16)) % (2**63 - 1)


@dataclass(frozen=True)
class Availability:
    available: list[str]
    excluded: dict[str, str]
    as_of: dict[str, AsOfItem]


def resolve_availability(impression: Impression, catalog: ItemCatalog) -> Availability:
    """只使用请求时刻已生效且已登记的属性版本；无可见版本视为不可用，不补齐业态。"""
    available: list[str] = []
    excluded: dict[str, str] = {}
    as_of: dict[str, AsOfItem] = {}
    for item in impression.candidates:
        try:
            snapshot = catalog.as_of(item, impression.requested_at)
        except NoVersionError:
            excluded[item] = "no_visible_version_at_request"
            continue
        as_of[item] = snapshot
        if not snapshot.open:
            excluded[item] = "closed_at_request"
            continue
        available.append(item)
    return Availability(available=available, excluded=excluded, as_of=as_of)


def count_legal_lists(available: list[str], as_of: dict[str, AsOfItem], k: int,
                      cap: int) -> int:
    """按业态分组的动态规划计数：不物化列表也能得到合法列表总数。"""
    if k <= 0:
        return 0
    groups: dict[str, int] = {}
    for item in available:
        category = as_of[item].category
        groups[category] = groups.get(category, 0) + 1
    dp = [1] + [0] * k
    for count in groups.values():
        nxt = [0] * (k + 1)
        for used in range(k + 1):
            if not dp[used]:
                continue
            for take in range(0, min(count, cap, k - used) + 1):
                nxt[used + take] += dp[used] * math.comb(count, take)
        dp = nxt
    return dp[k]


def enumerate_legal_lists(available: list[str], as_of: dict[str, AsOfItem], k: int,
                          cap: int) -> list[tuple[str, ...]]:
    if k <= 0:
        return []
    lists: list[tuple[str, ...]] = []
    for combo in itertools.combinations(available, k):
        counts: dict[str, int] = {}
        legal = True
        for item in combo:
            category = as_of[item].category
            counts[category] = counts.get(category, 0) + 1
            if counts[category] > cap:
                legal = False
                break
        if legal:
            lists.append(combo)
    return lists


def display_list(
    available: list[str], as_of: dict[str, AsOfItem], scores: dict[str, float],
    size: int, cap: int,
) -> list[str]:
    """按分数降序展示；同业态至多 cap 个；长度不足时如实变短。"""
    ordered = sorted(available, key=lambda item: (-scores[item], item))
    counts: dict[str, int] = {}
    chosen: list[str] = []
    for item in ordered:
        category = as_of[item].category
        if counts.get(category, 0) >= cap:
            continue
        counts[category] = counts.get(category, 0) + 1
        chosen.append(item)
        if len(chosen) == size:
            break
    return chosen


def _softmax(values: list[float], temperature: float) -> list[float]:
    if not values:
        return []
    peak = max(values)
    exps = [math.exp((value - peak) / temperature) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


@dataclass
class ListDecision:
    request_id: str
    requested_at: str
    requested_list_size: int
    available: list[str]
    excluded: dict[str, str]
    as_of: dict[str, AsOfItem]
    final_size: int
    display: list[str]
    fill: dict
    legal_lists: list[tuple[str, ...]]
    list_weights: list[float]
    randomized_action_probs: dict[str, float]
    deterministic_draw_probs: dict[str, float]
    estimate: dict = field(default_factory=dict)
    timing_seconds: float = 0.0

    def best_list(self) -> tuple[str, ...]:
        if not self.legal_lists:
            return tuple(self.display)
        order = sorted(range(len(self.legal_lists)),
                       key=lambda i: (-self.list_weights[i], self.legal_lists[i]))
        return self.legal_lists[order[0]]


def _fill_report(requested, listed, available, excluded, as_of, cap) -> dict:
    reasons = []
    if listed < requested:
        if not available:
            reasons.append("没有可用候选：全部闭店或缺少请求时刻可见的属性版本")
        else:
            counts: dict[str, int] = {}
            for item in available:
                category = as_of[item].category
                counts[category] = counts.get(category, 0) + 1
            capped = sum(min(n, cap) for n in counts.values())
            if capped < requested:
                over = {c: n for c, n in counts.items() if n > cap}
                if over:
                    reasons.append(
                        f"同业态上限 {cap}：{over}，最多可填 {capped} 项"
                    )
                else:
                    reasons.append(
                        f"可用候选仅 {len(available)} 个（被排除 {len(excluded)} 个），"
                        f"少于请求长度 {requested}"
                    )
            else:
                reasons.append("约束下的合法列表无法达到请求长度")
    return {
        "requested_list_size": requested,
        "available_count": len(available),
        "final_size": listed,
        "max_per_category": cap,
        "excluded": dict(excluded),
        "reasons": reasons,
    }


def build_list_policy(
    impression: Impression,
    catalog: ItemCatalog,
    scores: dict[str, float],
    list_size: int,
    max_per_category: int,
    temperature: float = 0.10,
    max_exact_lists: int = DEFAULT_EXACT_CAP,
    mc_samples: int = DEFAULT_MC_SAMPLES,
    seed: int = 0,
) -> ListDecision:
    started = time.perf_counter()
    availability = resolve_availability(impression, catalog)
    available, as_of, excluded = availability.available, availability.as_of, availability.excluded
    # 打分函数可能没有覆盖某个可用项目（例如合成演示）；缺失时用 0 分，排序稳定且可解释。
    scores = {item: scores.get(item, 0.0) for item in impression.candidates}

    shown = display_list(available, as_of, scores, list_size, max_per_category)
    k = len(shown)
    fill = _fill_report(list_size, k, available, excluded, as_of, max_per_category)

    deterministic_probs = (
        {item: (1.0 / k if item in shown else 0.0) for item in impression.candidates}
        if k else {item: 0.0 for item in impression.candidates}
    )

    if k == 0:
        decision = ListDecision(
            request_id=impression.request_id,
            requested_at=impression.requested_at.isoformat(),
            requested_list_size=list_size,
            available=available, excluded=excluded, as_of=as_of, final_size=0,
            display=[], fill=fill, legal_lists=[], list_weights=[],
            randomized_action_probs={item: 0.0 for item in impression.candidates},
            deterministic_draw_probs=deterministic_probs,
            estimate={"method": "empty_list", "legal_list_count": 0},
            timing_seconds=time.perf_counter() - started,
        )
        return decision

    exact_count = count_legal_lists(available, as_of, k, max_per_category)
    if exact_count <= max_exact_lists:
        legal = enumerate_legal_lists(available, as_of, k, max_per_category)
        values = [sum(scores[item] for item in combo) / len(combo) for combo in legal]
        weights = _softmax(values, temperature)
        randomized = {item: 0.0 for item in impression.candidates}
        for combo, weight in zip(legal, weights):
            share = weight / len(combo)
            for item in combo:
                randomized[item] += share
        estimate = {
            "path": "randomized_list_policy",
            "method": "exact_enumeration",
            "legal_list_count": exact_count,
            "enumerated_lists": len(legal),
            "estimated_ops": exact_count * max(k, 1),
            "monte_carlo_standard_error_max": 0.0,
            "action_prob_error": "0（精确枚举）",
        }
    else:
        rng = random.Random(stable_seed(seed, impression.request_id))
        sampled: list[tuple[str, ...]] = []
        attempts = 0
        limit = mc_samples * 500 + 2000
        while len(sampled) < mc_samples and attempts < limit:
            attempts += 1
            combo = tuple(rng.sample(available, k))
            counts: dict[str, int] = {}
            for item in combo:
                category = as_of[item].category
                counts[category] = counts.get(category, 0) + 1
            if max(counts.values()) <= max_per_category:
                sampled.append(combo)
        if not sampled:
            raise RuntimeError(f"{impression.request_id}: 约束拒绝采样未获得合法列表")
        values = [sum(scores[item] for item in combo) / len(combo) for combo in sampled]
        weights = _softmax(values, temperature)
        total = sum(weights)
        randomized = {item: 0.0 for item in impression.candidates}
        per_action: dict[str, list[float]] = {item: [] for item in impression.candidates}
        for combo, weight in zip(sampled, weights):
            share = weight / total / len(combo)
            for item in combo:
                per_action[item].append(share)
        stderr = {}
        for item, values_i in per_action.items():
            randomized[item] = sum(values_i)
            if len(values_i) > 1:
                mean = randomized[item]
                variance = sum((v - mean) ** 2 for v in values_i) / (len(values_i) - 1)
                stderr[item] = math.sqrt(variance / len(values_i))
            else:
                stderr[item] = 0.0
        estimate = {
            "path": "randomized_list_policy",
            "method": "monte_carlo",
            "legal_list_count": exact_count,
            "sampled_lists": len(sampled),
            "rejection_attempts": attempts,
            "acceptance_rate": len(sampled) / attempts if attempts else 0.0,
            "estimated_ops": exact_count * max(k, 1) + attempts * k,
            "monte_carlo_standard_error_max": max(stderr.values()),
            "monte_carlo_standard_error_by_action": stderr,
            "action_prob_error": "抽样误差：见各动作蒙特卡洛标准误；固定种子可复现",
        }
        legal = sampled

    decision = ListDecision(
        request_id=impression.request_id,
        requested_at=impression.requested_at.isoformat(),
        requested_list_size=list_size,
        available=available, excluded=excluded, as_of=as_of, final_size=k,
        display=shown, fill=fill, legal_lists=legal, list_weights=weights,
        randomized_action_probs=randomized,
        deterministic_draw_probs=deterministic_probs,
        estimate=estimate,
        timing_seconds=time.perf_counter() - started,
    )
    return decision


def decision_to_dict(decision: ListDecision, scores: dict[str, float]) -> dict:
    return {
        "request_id": decision.request_id,
        "requested_at": decision.requested_at,
        "display_list": list(decision.display),
        "best_legal_list": list(decision.best_list()),
        "available": decision.available,
        "excluded": decision.excluded,
        "item_categories": {item: decision.as_of[item].category for item in decision.as_of},
        "scores": {item: scores[item] for item in decision.available},
        "deterministic_draw_probs": decision.deterministic_draw_probs,
        "randomized_list_action_probs": decision.randomized_action_probs,
        "fill_shortfall": decision.fill,
        "computation": decision.estimate,
        "timing_seconds": decision.timing_seconds,
    }
