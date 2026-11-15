"""日志校验、去重、时间切分与属性 as-of 版本查询。

原则：
- 同 request_id 重复上报，内容完全一致才视为同一条；任何字段不同都拒绝。
- 切分按 requested_at 的请求级时间戳，同一请求的多行不可能跨集合（去重后每请求一行）。
- 项目属性只使用 request_timestamp <= requested_at 的最后一条 effective_at 版本；
  known_at > requested_at 的版本即使 effective_at 更早也不可见。
- 缺失（null）与 0 严格区分，未知字段保持 None，不补零。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .errors import (
    DuplicateConflictError,
    InputContractError,
    NoVersionError,
    UnknownItemError,
)
from .impression_reader import load

REQUIRED_IMPRESSION_FIELDS = (
    "request_id",
    "requested_at",
    "candidates",
    "selected_item",
    "logging_propensity",
    "reward_click",
    "reward_redeem",
)
REQUIRED_ITEM_FIELDS = ("item_key", "category", "effective_at", "known_at", "open")
REWARD_FIELDS = ("reward_click", "reward_redeem")


def parse_time(value, field: str) -> datetime:
    if not isinstance(value, str):
        raise InputContractError(f"{field} 必须是 ISO-8601 字符串，收到 {value!r}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise InputContractError(f"{field} 不是合法 ISO-8601 时间：{value!r}") from exc
    if parsed.tzinfo is None:
        raise InputContractError(f"{field} 必须带时区，收到 {value!r}")
    # 保留原始偏移：时段特征必须用请求当地小时，而不是被统一换算成 UTC 后的小时。
    return parsed


def _require_fields(record, fields, kind: str) -> None:
    missing = [name for name in fields if name not in record]
    if missing:
        raise InputContractError(f"{kind} 缺少字段 {missing}")


def _check_canonical_candidates(candidates, request_id: str) -> None:
    if not isinstance(candidates, list) or not candidates:
        raise InputContractError(f"{request_id}: candidates 必须是非空列表")
    if any(not isinstance(item, str) or not item for item in candidates):
        raise InputContractError(f"{request_id}: candidates 只能是非空字符串")
    if len(set(candidates)) != len(candidates):
        raise InputContractError(f"{request_id}: candidates 有重复项目，动作空间不明确")


@dataclass(frozen=True)
class ItemVersion:
    item_key: str
    category: str
    effective_at: datetime
    known_at: datetime
    open: bool


@dataclass(frozen=True)
class Impression:
    request_id: str
    requested_at: datetime
    candidates: tuple[str, ...]
    selected_item: str
    logging_propensity: float
    reward_click: int
    reward_redeem: int

    @property
    def time_bucket(self) -> str:
        return self.requested_at.strftime("%Y-%m-%d %H:%M")

    @property
    def local_hour(self) -> int:
        return self.requested_at.hour


@dataclass(frozen=True)
class AsOfItem:
    item_key: str
    category: str | None
    open: bool | None
    version_time: datetime | None
    known_at: datetime | None


class ItemCatalog:
    def __init__(self, versions: list[ItemVersion]):
        self._by_key: dict[str, list[ItemVersion]] = {}
        for version in versions:
            self._by_key.setdefault(version.item_key, []).append(version)
        for key in self._by_key:
            self._by_key[key].sort(key=lambda v: (v.effective_at, v.known_at))

    @property
    def keys(self) -> set[str]:
        return set(self._by_key)

    def versions(self) -> list[ItemVersion]:
        return [version for versions in self._by_key.values() for version in versions]

    def as_of(self, item_key: str, at: datetime) -> AsOfItem:
        """请求时刻已经生效、且当时已登记的最新版本。"""
        if item_key not in self._by_key:
            raise UnknownItemError(f"items 缺少项目 {item_key!r}")
        visible = [v for v in self._by_key[item_key] if v.effective_at <= at and v.known_at <= at]
        if not visible:
            raise NoVersionError(
                f"项目 {item_key!r} 在 {at.isoformat()} 没有可用属性版本 "
                f"（后续版本 known_at 或 effective_at 晚于请求时间）"
            )
        latest = visible[-1]
        return AsOfItem(
            item_key=item_key,
            category=latest.category,
            open=latest.open,
            version_time=latest.effective_at,
            known_at=latest.known_at,
        )


def _normalize_impression(record) -> Impression:
    _require_fields(record, REQUIRED_IMPRESSION_FIELDS, "曝光记录")
    request_id = record["request_id"]
    if not isinstance(request_id, str) or not request_id:
        raise InputContractError(f"request_id 必须是非空字符串：{request_id!r}")
    requested_at = parse_time(record["requested_at"], f"{request_id}.requested_at")
    candidates = record["candidates"]
    _check_canonical_candidates(candidates, request_id)
    selected = record["selected_item"]
    if selected not in candidates:
        raise InputContractError(f"{request_id}: selected_item={selected!r} 不在候选集中")
    propensity = record["logging_propensity"]
    if isinstance(propensity, bool) or not isinstance(propensity, (int, float)):
        raise InputContractError(f"{request_id}: logging_propensity 必须是数值")
    propensity = float(propensity)
    if not 0.0 <= propensity <= 1.0:
        raise InputContractError(f"{request_id}: logging_propensity 必须在 [0,1]，收到 {propensity}")
    for field in REWARD_FIELDS:
        value = record[field]
        if value not in (0, 1):
            raise InputContractError(f"{request_id}: {field} 必须是 0 或 1，收到 {value!r}")
    return Impression(
        request_id=request_id,
        requested_at=requested_at,
        candidates=tuple(candidates),
        selected_item=selected,
        logging_propensity=propensity,
        reward_click=record["reward_click"],
        reward_redeem=record["reward_redeem"],
    )


def deduplicate(raw_rows: list[dict]) -> tuple[list[Impression], int]:
    """返回 (去重后的记录, 被移除的重复行数)。冲突即拒绝。"""
    seen: dict[str, Impression] = {}
    duplicates = 0
    for row in raw_rows:
        impression = _normalize_impression(row)
        previous = seen.get(impression.request_id)
        if previous is None:
            seen[impression.request_id] = impression
            continue
        if previous != impression:
            raise DuplicateConflictError(
                f"request_id={impression.request_id!r} 重复上报但内容冲突：\n"
                f"  先到={previous}\n  后到={impression}"
            )
        duplicates += 1
    ordered = sorted(seen.values(), key=lambda i: (i.requested_at, i.request_id))
    return ordered, duplicates


def _normalize_item(raw) -> ItemVersion:
    _require_fields(raw, REQUIRED_ITEM_FIELDS, "项目属性记录")
    key = raw["item_key"]
    if not isinstance(key, str) or not key:
        raise InputContractError(f"item_key 必须是非空字符串：{key!r}")
    if not isinstance(raw["category"], str) or not raw["category"]:
        raise InputContractError(f"{key}: category 必须是非空字符串")
    if not isinstance(raw["open"], bool):
        raise InputContractError(f"{key}: open 必须是布尔值")
    effective_at = parse_time(raw["effective_at"], f"{key}.effective_at")
    known_at = parse_time(raw["known_at"], f"{key}.known_at")
    return ItemVersion(
        item_key=key,
        category=raw["category"],
        effective_at=effective_at,
        known_at=known_at,
        open=raw["open"],
    )


def load_catalog(raw_rows: list[dict]) -> ItemCatalog:
    versions = [_normalize_item(row) for row in raw_rows]
    catalog = ItemCatalog(versions)
    # 同一 (item, effective_at) 出现两个不同版本时无法确定当时事实，拒绝。
    for key in catalog.keys:
        by_effective: dict[datetime, ItemVersion] = {}
        for version in catalog._by_key[key]:  # noqa: SLF001 - 同模块内检查
            if version.effective_at in by_effective and by_effective[version.effective_at] != version:
                raise InputContractError(
                    f"{key}: effective_at={version.effective_at.isoformat()} 存在冲突的属性版本"
                )
            by_effective[version.effective_at] = version
    return catalog


def check_item_references(impressions: list[Impression], catalog: ItemCatalog) -> None:
    missing = sorted(
        {item for imp in impressions for item in imp.candidates if item not in catalog.keys}
    )
    if missing:
        raise UnknownItemError(f"日志引用但 items 未登记的项目：{missing}")


@dataclass(frozen=True)
class Splits:
    train: list[Impression]
    validation: list[Impression]
    test: list[Impression]

    def as_dict(self) -> dict[str, list[Impression]]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


def make_splits(impressions: list[Impression], train_end: datetime,
                validation_end: datetime, test_end: datetime) -> Splits:
    """requested_at < train_end 训练；[train_end, validation_end) 验证；
    [validation_end, test_end) 测试；>= test_end 不进入任何集合（未来审计留出）。"""
    if not (train_end <= validation_end <= test_end):
        raise InputContractError("切分边界必须满足 train_end <= validation_end <= test_end")
    train, validation, test = [], [], []
    for imp in impressions:
        if imp.requested_at < train_end:
            train.append(imp)
        elif imp.requested_at < validation_end:
            validation.append(imp)
        elif imp.requested_at < test_end:
            test.append(imp)
    return Splits(train, validation, test)


def split_overlap_report(splits: Splits) -> dict:
    """按 request_id 复核集合互斥，并给出时间边界上的记录计数。"""
    names = {"train": splits.train, "validation": splits.validation, "test": splits.test}
    ids = {name: {imp.request_id for imp in rows} for name, rows in names.items()}
    overlaps = {}
    ordered = ["train", "validation", "test"]
    for i, left in enumerate(ordered):
        for right in ordered[i + 1:]:
            common = ids[left] & ids[right]
            if common:
                overlaps[f"{left}&{right}"] = sorted(common)
    times = {
        name: {
            "rows": len(rows),
            "min_requested_at": min((i.requested_at for i in rows), default=None),
            "max_requested_at": max((i.requested_at for i in rows), default=None),
        }
        for name, rows in names.items()
    }
    return {"overlaps": overlaps, "rows": times}


def load_fixture_dir(root: Path) -> tuple[list[dict], list[dict], dict]:
    root = Path(root)
    impressions_path = root / "impressions.jsonl"
    items_path = root / "items.jsonl"
    policies_path = root / "policies.json"
    for path in (impressions_path, items_path, policies_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if impressions_path.suffix != ".jsonl" or items_path.suffix != ".jsonl":
        raise InputContractError("曝光与项目资料必须为 .jsonl")
    raw_impressions = load(impressions_path)
    raw_items = load(items_path)
    policies = load(policies_path)
    if not isinstance(raw_impressions, list) or not isinstance(raw_items, list) or not isinstance(policies, dict):
        raise InputContractError("资料形状不符合约定（impressions/items 应为列表，policies 应为对象）")
    return raw_impressions, raw_items, policies


def read_policies(policies: dict) -> dict:
    required = (
        "hand_calculation_requests",
        "target_selected_action_probabilities",
        "train_end",
        "validation_end",
        "test_end",
        "list_size",
        "max_per_category",
        "bootstrap_repeats",
        "seed",
    )
    missing = [key for key in required if key not in policies]
    if missing:
        raise InputContractError(f"policies.json 缺少字段 {missing}")
    return {
        "hand_calculation_requests": list(policies["hand_calculation_requests"]),
        "target_probs": dict(policies["target_selected_action_probabilities"]),
        "train_end": parse_time(policies["train_end"], "train_end"),
        "validation_end": parse_time(policies["validation_end"], "validation_end"),
        "test_end": parse_time(policies["test_end"], "test_end"),
        "list_size": int(policies["list_size"]),
        "max_per_category": int(policies["max_per_category"]),
        "bootstrap_repeats": int(policies["bootstrap_repeats"]),
        "seed": int(policies["seed"]),
    }


def dumps_time(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()
