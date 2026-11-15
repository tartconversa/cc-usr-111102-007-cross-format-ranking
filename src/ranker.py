"""可使用业态与时段特征的可训练学习排序器（纯标准库逻辑回归）。

特征（全部在请求时刻可观测，且只从训练集统计）：
- 截距（全局基准）
- 业态 one-hot：候选在 as-of 时刻的 category
- 请求当地小时所在时段 one-hot：morning / afternoon / evening
- 项目训练点击率与训练核销率（向先验收缩；训练集从未出现的项目取中性先验 0.5）
- log1p(项目训练曝光量)：用连续量避免与截距完全共线；冷启动项目为 0

训练目标：观测选中项目的 reward_click（0/1），样本权重 = 1/logging_propensity。
热启动系数清零，L2 正则，全批量梯度下降。
为防止目标泄漏，每条训练行的项目率特征都会减去该行自身贡献（留一法）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .data import ItemCatalog, Impression, AsOfItem

HOUR_BUCKETS = (
    ("morning", range(0, 12)),
    ("afternoon", range(12, 18)),
    ("evening", range(18, 24)),
)


def hour_bucket(hour: int) -> str:
    for name, hours in HOUR_BUCKETS:
        if hour in hours:
            return name
    raise ValueError(hour)


def _neutral_prior() -> float:
    return 0.5


@dataclass
class ItemStats:
    """训练集中该项目的记录计数。views 实为“被记录为选中动作的次数”，不是曝光次数。"""

    clicks: int = 0
    logged_selections: int = 0
    redeems: int = 0


@dataclass
class Encoder:
    prior_rate: float
    smoothing: float
    categories: list[str]
    item_stats: dict[str, ItemStats] = field(default_factory=dict)

    def feature_names(self) -> list[str]:
        names = ["intercept"]
        names += [f"category={c}" for c in self.categories]
        names += [f"hour={name}" for name, _ in HOUR_BUCKETS]
        names += ["item_click_rate_train", "item_redeem_rate_train", "log1p_train_selections"]
        return names

    def encode(self, snapshot: AsOfItem, hour: int,
               subtract: ItemStats | None = None) -> list[float]:
        """subtract 供训练时减去当前行自身贡献，避免目标泄漏。"""
        vector = [1.0]
        category = snapshot.category
        vector += [1.0 if category == known else 0.0 for known in self.categories]
        bucket = hour_bucket(hour)
        vector += [1.0 if bucket == name else 0.0 for name, _ in HOUR_BUCKETS]
        stats = self.item_stats.get(snapshot.item_key)
        selections = 0 if stats is None else stats.logged_selections
        clicks = 0 if stats is None else stats.clicks
        redeems = 0 if stats is None else stats.redeems
        if subtract is not None:
            selections -= subtract.logged_selections
            clicks -= subtract.clicks
            redeems -= subtract.redeems
        if selections <= 0:
            click_rate = _neutral_prior()
            redeem_rate = _neutral_prior()
            selection_count = 0
        else:
            click_rate = (clicks + self.smoothing * self.prior_rate) / (
                selections + self.smoothing
            )
            redeem_rate = (redeems + self.smoothing * self.prior_rate) / (
                selections + self.smoothing
            )
            selection_count = selections
        vector += [click_rate, redeem_rate, math.log1p(selection_count)]
        return vector


def fit_encoder(train: list[Impression], catalog: ItemCatalog, smoothing: float = 2.0) -> Encoder:
    """只用训练集；业态集合与项目计数都不含验证/测试请求。"""
    categories = sorted({catalog.as_of(item, imp.requested_at).category
                         for imp in train for item in imp.candidates})
    item_stats: dict[str, ItemStats] = {}
    clicks = sum(imp.reward_click for imp in train)
    prior_rate = (clicks + 1.0) / (len(train) + 2.0)
    for imp in train:
        stats = item_stats.setdefault(imp.selected_item, ItemStats())
        stats.logged_selections += 1
        stats.clicks += imp.reward_click
        stats.redeems += imp.reward_redeem
    return Encoder(
        prior_rate=prior_rate,
        smoothing=smoothing,
        categories=categories,
        item_stats=item_stats,
    )


@dataclass
class LogisticRanker:
    encoder: Encoder
    weights: list[float]
    names: list[str]
    training: dict

    def score_vector(self, vector: list[float]) -> float:
        z = sum(w * x for w, x in zip(self.weights, vector))
        return _sigmoid(z)

    def score(self, snapshot: AsOfItem, hour: int) -> float:
        return self.score_vector(self.encoder.encode(snapshot, hour))

    def explain(self, snapshot: AsOfItem, hour: int) -> dict:
        vector = self.encoder.encode(snapshot, hour)
        contributions = {
            name: weight * value
            for name, weight, value in zip(self.names, self.weights, vector)
            if abs(weight * value) > 1e-12
        }
        return {
            "item": snapshot.item_key,
            "category": snapshot.category,
            "hour_bucket": hour_bucket(hour),
            "score": self.score(snapshot, hour),
            "contributions": contributions,
        }


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def _training_rows(train, catalog, encoder, reward, subtract_self: bool):
    rows = []
    for imp in train:
        snapshot = catalog.as_of(imp.selected_item, imp.requested_at)
        own = ItemStats(logged_selections=1, clicks=imp.reward_click,
                        redeems=imp.reward_redeem) if subtract_self else None
        vector = encoder.encode(snapshot, imp.requested_at.hour, subtract=own)
        rows.append((vector, float(getattr(imp, reward)), 1.0 / imp.logging_propensity))
    return rows


def _standardize(rows, width):
    means = [0.0] * width
    scales = [1.0] * width
    for index in range(width):
        values = [vector[index] for vector, _, _ in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        means[index] = mean
        scales[index] = math.sqrt(variance) if variance > 1e-12 else 1.0
    normalized = [
        ([(value - means[i]) / scales[i] for i, value in enumerate(vector)], target, weight)
        for vector, target, weight in rows
    ]
    return normalized, means, scales


def _fit_normalized(normalized, width, epochs, learning_rate, l2):
    weights = [0.0] * width
    total_weight = sum(weight for _, _, weight in normalized) or 1.0
    curve = []
    for epoch in range(epochs):
        gradient = [0.0] * width
        loss = 0.0
        for vector, target, weight in normalized:
            prediction = _sigmoid(sum(w * x for w, x in zip(weights, vector)))
            error = (prediction - target) * weight
            for index, value in enumerate(vector):
                gradient[index] += error * value
            p = min(max(prediction, 1e-9), 1 - 1e-9)
            loss -= weight * (target * math.log(p) + (1 - target) * math.log(1 - p))
        for index in range(width):
            weights[index] -= learning_rate * (gradient[index] / total_weight + l2 * weights[index])
        if epoch % 50 == 0 or epoch == epochs - 1:
            curve.append(round(loss / total_weight, 6))
    return weights, curve


def _weighted_log_loss(model: LogisticRanker, impressions, catalog, reward) -> float:
    total = 0.0
    weight_sum = 0.0
    for imp in impressions:
        snapshot = catalog.as_of(imp.selected_item, imp.requested_at)
        prediction = model.score(snapshot, imp.requested_at.hour)
        p = min(max(prediction, 1e-9), 1 - 1e-9)
        target = float(getattr(imp, reward))
        weight = 1.0 / imp.logging_propensity
        total -= weight * (target * math.log(p) + (1 - target) * math.log(1 - p))
        weight_sum += weight
    return total / weight_sum if weight_sum else float("nan")


def train_ranker(
    train: list[Impression],
    catalog: ItemCatalog,
    epochs: int = 300,
    learning_rate: float = 0.25,
    l2: float = 0.01,
    reward: str = "reward_click",
) -> LogisticRanker:
    if not train:
        raise ValueError("训练集为空，无法拟合排序器")
    encoder = fit_encoder(train, catalog)
    names = encoder.feature_names()
    width = len(names)
    rows = _training_rows(train, catalog, encoder, reward, subtract_self=True)
    normalized, means, scales = _standardize(rows, width)
    weights, curve = _fit_normalized(normalized, width, epochs, learning_rate, l2)
    # 换回原始特征空间：w_raw_i = w_i / scale_i；截距吸收各特征的均值项。
    raw = [weights[i] / scales[i] for i in range(width)]
    raw[0] -= sum(weights[i] * means[i] / scales[i] for i in range(1, width))
    return LogisticRanker(
        encoder=encoder,
        weights=raw,
        names=names,
        training={
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2": l2,
            "sample_count": len(train),
            "weighted_loss_curve": curve,
            "reward": reward,
            "target_leakage_guard": "项目率特征留一法剔除当前行；均值/方差只来自训练集",
        },
    )


def tune_ranker(
    train: list[Impression],
    validation: list[Impression],
    catalog: ItemCatalog,
    grid: list[dict] | None = None,
    reward: str = "reward_click",
) -> tuple[LogisticRanker, dict]:
    """只在验证集上选择超参数；测试集在调参期间不参与。"""
    if not validation:
        raise ValueError("验证集为空：无法进行不触及测试集的调参")
    grid = grid or [
        {"learning_rate": 0.1, "l2": 0.01, "epochs": 200},
        {"learning_rate": 0.25, "l2": 0.01, "epochs": 300},
        {"learning_rate": 0.25, "l2": 0.05, "epochs": 300},
        {"learning_rate": 0.5, "l2": 0.01, "epochs": 400},
    ]
    candidates = []
    best = None
    for spec in grid:
        model = train_ranker(train, catalog, reward=reward, **spec)
        loss = _weighted_log_loss(model, validation, catalog, reward)
        candidates.append({**spec, "validation_weighted_log_loss": loss})
        if best is None or loss < best[1]:
            best = (model, loss, spec)
    model, loss, spec = best
    return model, {
        "objective": "validation 上的 1/propensity 加权对数损失（越低越好）",
        "grid": candidates,
        "selected": spec,
        "selected_validation_loss": loss,
        "test_used_for_tuning": False,
        "note": "验证集来自 train_end 与 validation_end 之间的请求，与测试集按请求时间互斥。",
    }


def ranker_scores(
    ranker: LogisticRanker, impression: Impression, catalog: ItemCatalog
) -> dict[str, float]:
    """只给请求时刻有可见版本的候选打分；无版本的项目交由策略层标记为不可用。"""
    from .errors import NoVersionError

    scores = {}
    for item in impression.candidates:
        try:
            snapshot = catalog.as_of(item, impression.requested_at)
        except NoVersionError:
            continue
        scores[item] = ranker.score(snapshot, impression.requested_at.hour)
    return scores


def cold_start_template(ranker: LogisticRanker, category: str,
                        known_hours: list[int]) -> dict[str, float]:
    """冷启动项目的分数只依赖业态与时段；曝光/点击率特征取中性值。

    known_hours 必须来自训练集观测到的请求小时，不能拿测试/未来小时回填。
    """
    vector = [1.0]
    vector += [1.0 if category == known else 0.0 for known in ranker.encoder.categories]
    for name, hours in HOUR_BUCKETS:
        # 该时段在训练集里是否出现过；未出现时 one-hot 全 0。
        appears = any(hour in hours for hour in known_hours)
        vector.append(1.0 if appears else 0.0)
    vector += [_neutral_prior(), _neutral_prior(), 0.0]
    return {
        "score": ranker.score_vector(vector),
        "given_category": category,
        "category_known_in_training": category in ranker.encoder.categories,
    }


def baseline_scores_for_request(baseline, impression: Impression, catalog: ItemCatalog) -> dict[str, float]:
    """基线只按项目训练统计打分；请求时刻只决定候选可见性，不改变分数。"""
    from .errors import NoVersionError

    scores = {}
    for item in impression.candidates:
        try:
            catalog.as_of(item, impression.requested_at)
        except NoVersionError:
            continue
        scores[item] = baseline.score(item)
    return scores
