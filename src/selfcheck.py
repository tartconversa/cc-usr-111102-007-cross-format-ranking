"""对流水线输出做离线自检，返回结构化结论。"""

from __future__ import annotations


def run_self_check(payload: dict) -> dict:
    checks = []

    def check(name, condition, detail):
        checks.append({"check": name, "passed": bool(condition), "detail": detail})

    hand = payload["hand_check"]["assertions"]
    check("手算 IPS=0.25", hand["ips_ok"], f"actual={hand['ips_actual']}")
    check("手算 SNIPS=1/3", hand["snips_ok"], f"actual={hand['snips_actual']}")

    audit = payload["time_leakage_audit"]
    check("切分按请求互斥", audit["overlap_free"], audit["split_membership_overlaps"])
    check("请求时间顺序正确", audit["time_ordering_ok"], audit["bounds"])
    check("无属性版本穿越", not audit["attribute_version_violations"],
          audit["attribute_version_violations"])

    unsupported = payload["policy_scenarios"]["unsupported_new_item_target"]["support"]
    check("不可支持策略被识别", unsupported["status"] == "proven_zero_support"
          and not unsupported["full_policy_value_identifiable"], unsupported["status"])

    uniform = payload["policy_scenarios"]["uniform_random_target"]["support"]
    check("全支持目标可识别", uniform["full_policy_value_identifiable"], uniform["status"])

    for name, entry in payload["test_evaluation"].items():
        plain = entry["unclipped"]
        clipped = entry.get("clipped", {})
        check(f"{name} 裁剪结果另列",
              "clipped" in entry and "bias" in clipped, clipped.get("clip_threshold"))
        check(f"{name} 支持状态已声明", bool(entry["support"]["statement"]),
              entry["support"]["status"])
    check("动作概率含重排与最终抽取",
          _all_probabilities_normalized(payload["test_list_examples"]),
          "每个请求的 deterministic_draw_probs 之和为 1")
    check("报告不称线上因果", _claims_disclaim(payload), payload["claims"][0])

    version = payload["item_version_demo"]
    check("属性版本切换生效",
          version["before_2026-10-20"]["open"] is True
          and version["from_2026-10-20"]["open"] is False,
          {"before": version["before_2026-10-20"]["open"],
           "after": version["from_2026-10-20"]["open"]})

    fill = payload["fill_shortfall_demo"]
    check("无法填满给出原因", bool(fill["fill"]["reasons"]) and
          fill["fill"]["final_size"] < fill["request"]["requested_list_size"],
          fill["fill"]["reasons"])

    computation = payload["computation_design"]
    check("枚举/抽样误差已说明",
          "exact_enumeration" in computation["methods"] or
          "monte_carlo" in computation["methods"],
          list(computation["methods"]))

    return {
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }


def _all_probabilities_normalized(examples: list[dict]) -> bool:
    for example in examples:
        probs = example.get("deterministic_draw_probs", {})
        if probs and abs(sum(probs.values()) - 1.0) > 1e-9:
            return False
        randomized = example.get("randomized_list_action_probs", {})
        if randomized and abs(sum(randomized.values()) - 1.0) > 1e-6:
            return False
    return True


def _claims_disclaim(payload: dict) -> bool:
    joined = " ".join(payload["claims"])
    return "不是线上因果收益的证明" in joined
