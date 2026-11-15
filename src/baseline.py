"""可解释流行度基线。定义在项目级动作上，输出每个候选的点击倾向分数。

基线只依赖观测日志，不使用未来信息：
- 训练集上每个项目的拉普拉斯平滑点击率
  (点击次数 + 先验*伪计数) / (被记录为选中动作的次数 + 伪计数)；
- 训练集从未出现的项目使用先验 1/2（而不是 0），并显式标记为冷启动；
- 分数是排序依据，不是校准后的概率；下游用单调 softmax 把它变成列表策略的动作分布。

注意：日志每请求只记录一个被选动作，因此“选中次数”是记录策略下的选择次数，
不是曝光次数。在候选集均匀记录时，它与曝光成正比，但仍按选择次数命名。
"""

from __future__ import annotations

from dataclasses import dataclass

from .data import Impression


@dataclass(frozen=True)
class PopularityBaseline:
    prior_mean: float
    pseudo_views: int
    clicks: dict[str, int]
    logged_selections: dict[str, int]
    reward: str = "reward_click"

    def score_items(self, item_keys: list[str]) -> dict[str, float]:
        return {key: self.score(key) for key in item_keys}

    def score(self, item_key: str) -> float:
        return (
            self.clicks.get(item_key, 0) + self.prior_mean * self.pseudo_views
        ) / (self.logged_selections.get(item_key, 0) + self.pseudo_views)

    def explain(self, item_key: str) -> dict:
        clicks = self.clicks.get(item_key, 0)
        selections = self.logged_selections.get(item_key, 0)
        return {
            "item": item_key,
            "train_clicks": clicks,
            "train_logged_selections": selections,
            "cold_start": selections == 0,
            "prior": self.prior_mean,
            "score": self.score(item_key),
            "formula": f"({clicks} + {self.prior_mean}*{self.pseudo_views}) / "
                       f"({selections} + {self.pseudo_views})",
        }


def train_popularity_baseline(
    train: list[Impression], reward: str = "reward_click", pseudo_views: int = 2
) -> PopularityBaseline:
    if not train:
        raise ValueError("训练集为空，无法拟合流行度基线")
    clicks: dict[str, int] = {}
    selections: dict[str, int] = {}
    outcome = 0
    for impression in train:
        selections[impression.selected_item] = selections.get(impression.selected_item, 0) + 1
        value = getattr(impression, reward)
        outcome += value
        if value:
            clicks[impression.selected_item] = clicks.get(impression.selected_item, 0) + 1
    # 伪计数下的先验均值：训练集整体奖励率，向 0.5 收缩，避免极端值。
    prior_mean = (outcome + 1.0) / (len(train) + 2.0)
    return PopularityBaseline(
        prior_mean=prior_mean,
        pseudo_views=pseudo_views,
        clicks=clicks,
        logged_selections=selections,
        reward=reward,
    )


def popularity_weights(baseline: PopularityBaseline, item_keys: list[str]) -> list[float]:
    """单调变换：分数 -> 权重。softmax 温度固定，便于复算；不改变排序。"""
    import math

    scores = [baseline.score(key) for key in item_keys]
    peak = max(scores)
    exps = [math.exp((score - peak) / 0.25) for score in scores]
    total = sum(exps)
    return [value / total for value in exps]
