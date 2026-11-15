import math
import tempfile
import unittest
from pathlib import Path

from src import ope
from src.baseline import train_popularity_baseline
from src.data import (
    deduplicate,
    load_catalog,
    load_fixture_dir,
    make_splits,
    parse_time,
    read_policies,
)
from src.errors import DuplicateConflictError, InputContractError, NoVersionError
from src.evaluation import run_pipeline
from src.policy import (
    build_list_policy,
    count_legal_lists,
    enumerate_legal_lists,
    resolve_availability,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def impression(**overrides):
    base = dict(
        request_id="t1",
        requested_at="2026-10-01T10:00:00+08:00",
        candidates=["a", "b"],
        selected_item="a",
        logging_propensity=0.5,
        reward_click=1,
        reward_redeem=0,
    )
    base.update(overrides)
    return base


def catalog_rows():
    return [
        {"item_key": "a", "category": "dining",
         "effective_at": "2026-09-01T00:00:00+08:00",
         "known_at": "2026-09-01T00:00:00+08:00", "open": True},
        {"item_key": "b", "category": "dining",
         "effective_at": "2026-09-01T00:00:00+08:00",
         "known_at": "2026-09-01T00:00:00+08:00", "open": True},
        {"item_key": "c", "category": "family",
         "effective_at": "2026-09-01T00:00:00+08:00",
         "known_at": "2026-09-01T00:00:00+08:00", "open": True},
        {"item_key": "d", "category": "dining",
         "effective_at": "2026-09-01T00:00:00+08:00",
         "known_at": "2026-09-01T00:00:00+08:00", "open": True},
    ]


def one(rows=None):
    return deduplicate(rows or [impression()])[0][0]


class InputContractTests(unittest.TestCase):
    def test_identical_duplicate_is_deduplicated(self):
        impressions, removed = deduplicate([impression(), impression()])
        self.assertEqual(len(impressions), 1)
        self.assertEqual(removed, 1)

    def test_conflicting_duplicate_is_rejected(self):
        with self.assertRaises(DuplicateConflictError):
            deduplicate([impression(), impression(reward_click=0)])

    def test_propensity_out_of_range_is_rejected(self):
        with self.assertRaises(InputContractError):
            deduplicate([impression(logging_propensity=1.5)])

    def test_selected_item_must_be_candidate(self):
        with self.assertRaises(InputContractError):
            deduplicate([impression(selected_item="z")])

    def test_zero_propensity_is_allowed_and_flagged(self):
        impressions, _ = deduplicate([impression(logging_propensity=0.0)])
        rows = ope.per_request_rows(
            impressions, "reward_click",
            lambda imp: {imp.selected_item: 1.0}, False,
        )
        report = ope.support_report(rows, False)
        self.assertTrue(math.isinf(rows[0].weight))
        self.assertEqual(report["status"], "proven_zero_support")

    def test_missing_field_is_rejected(self):
        row = impression()
        del row["logging_propensity"]
        with self.assertRaises(InputContractError):
            deduplicate([row])


class SplitTests(unittest.TestCase):
    def setUp(self):
        raw_impressions, _, raw_policies = load_fixture_dir(FIXTURES)
        self.cfg = read_policies(raw_policies)
        self.impressions, _ = deduplicate(raw_impressions)

    def test_splits_are_disjoint_and_time_ordered(self):
        splits = make_splits(self.impressions, self.cfg["train_end"],
                             self.cfg["validation_end"], self.cfg["test_end"])
        train = {i.request_id for i in splits.train}
        val = {i.request_id for i in splits.validation}
        test = {i.request_id for i in splits.test}
        self.assertFalse(train & val or train & test or val & test)
        self.assertLess(max(i.requested_at for i in splits.train),
                        min(i.requested_at for i in splits.validation))
        self.assertLess(max(i.requested_at for i in splits.validation),
                        min(i.requested_at for i in splits.test))

    def test_each_request_appears_once(self):
        splits = make_splits(self.impressions, self.cfg["train_end"],
                             self.cfg["validation_end"], self.cfg["test_end"])
        ids = [i.request_id for rows in splits.as_dict().values() for i in rows]
        self.assertEqual(len(ids), len(set(ids)))


class HandCalculationTests(unittest.TestCase):
    def test_hand_calculation_matches_required_values(self):
        raw_impressions, _, raw_policies = load_fixture_dir(FIXTURES)
        cfg = read_policies(raw_policies)
        impressions, _ = deduplicate(raw_impressions)
        by_id = {i.request_id: i for i in impressions}
        manual = cfg["target_probs"]
        rows = ope.per_request_rows(
            [by_id[r] for r in cfg["hand_calculation_requests"]],
            "reward_click",
            lambda imp: {imp.selected_item: manual[imp.request_id]},
            uniform_verified=True,
        )
        check = ope.hand_check(rows)
        self.assertAlmostEqual(check["ips"], 0.25, places=12)
        self.assertAlmostEqual(check["snips"], 1 / 3, places=12)
        self.assertAlmostEqual(check["ess"], 1.8, places=12)


class SupportTests(unittest.TestCase):
    def test_action_outside_candidates_is_proven_zero_support(self):
        rows = ope.per_request_rows(
            deduplicate([impression()])[0], "reward_click",
            lambda imp: {"never-shown": 1.0}, True,
        )
        report = ope.support_report(rows, True)
        self.assertEqual(report["status"], "proven_zero_support")
        self.assertFalse(report["full_policy_value_identifiable"])

    def test_uniform_verified_full_support_on_candidates(self):
        rows = ope.per_request_rows(
            deduplicate([impression()])[0], "reward_click",
            lambda imp: {item: 0.5 for item in imp.candidates}, True,
        )
        report = ope.support_report(rows, True)
        self.assertEqual(report["status"], "verified_full_support")
        self.assertTrue(report["full_policy_value_identifiable"])

    def test_unverified_policy_marks_target_on_unlogged_action_unidentifiable(self):
        rows = ope.per_request_rows(
            deduplicate([impression()])[0], "reward_click",
            lambda imp: {"b": 1.0}, False,
        )
        report = ope.support_report(rows, False)
        self.assertFalse(report["full_policy_value_identifiable"])
        self.assertEqual(report["status"], "support_unverified")

    def test_target_on_logged_action_has_known_support_even_if_unverified(self):
        rows = ope.per_request_rows(
            deduplicate([impression()])[0], "reward_click",
            lambda imp: {imp.selected_item: 1.0}, False,
        )
        report = ope.support_report(rows, False)
        self.assertTrue(report["full_policy_value_identifiable"])

    def test_clip_is_reported_separately_with_bias_label(self):
        rows = ope.per_request_rows(
            deduplicate([impression()])[0], "reward_click",
            lambda imp: {item: 0.5 for item in imp.candidates}, True,
        )
        result = ope.evaluate_policy(
            deduplicate([impression()])[0], "reward_click",
            lambda imp: {item: 0.5 for item in imp.candidates},
            repeats=10, seed=1, rng_factory=lambda s: __import__("random").Random(s),
            uniform_verified=True, clip=0.5,
        )
        self.assertIn("clipped", result)
        self.assertIn("有偏", result["clipped"]["bias"])


class PolicyConstraintTests(unittest.TestCase):
    def setUp(self):
        self.catalog = load_catalog(catalog_rows())
        self.imp = one([impression(candidates=["a", "b", "c"], selected_item="a")])

    def test_draw_probabilities_include_final_draw(self):
        decision = build_list_policy(
            self.imp, self.catalog, {"a": 0.9, "b": 0.8, "c": 0.7},
            list_size=2, max_per_category=1,
        )
        self.assertEqual(decision.display, ["a", "c"])
        self.assertAlmostEqual(sum(decision.deterministic_draw_probs.values()), 1.0)
        self.assertAlmostEqual(decision.deterministic_draw_probs["a"], 0.5)
        self.assertEqual(decision.deterministic_draw_probs["b"], 0.0)

    def test_category_cap_shortens_list_with_reason(self):
        imp = one([impression(candidates=["a", "b", "d"], selected_item="a")])
        decision = build_list_policy(
            imp, self.catalog, {"a": 0.9, "b": 0.8, "d": 0.1},
            list_size=3, max_per_category=2,
        )
        self.assertEqual(decision.display, ["a", "b"])
        self.assertTrue(decision.fill["reasons"])

    def test_closed_item_excluded_with_reason(self):
        rows = [{"item_key": "a", "category": "dining",
                 "effective_at": "2026-09-01T00:00:00+08:00",
                 "known_at": "2026-09-01T00:00:00+08:00", "open": False}]
        catalog = load_catalog(rows)
        imp = one([impression(candidates=["a"], selected_item="a")])
        decision = build_list_policy(imp, catalog, {"a": 1.0}, 3, 2)
        self.assertEqual(decision.display, [])
        self.assertEqual(decision.excluded["a"], "closed_at_request")
        self.assertTrue(decision.fill["reasons"])

    def test_legal_list_count_matches_enumeration(self):
        availability = resolve_availability(self.imp, self.catalog)
        for k in (1, 2, 3):
            counted = count_legal_lists(availability.available, availability.as_of, k, 2)
            enumerated = enumerate_legal_lists(
                availability.available, availability.as_of, k, 2)
            self.assertEqual(counted, len(enumerated))

    def test_monte_carlo_reproducible_and_reports_error(self):
        scores = {"a": 0.9, "b": 0.8, "c": 0.7}
        first = build_list_policy(self.imp, self.catalog, scores, 2, 2,
                                  max_exact_lists=1, mc_samples=500, seed=7)
        second = build_list_policy(self.imp, self.catalog, scores, 2, 2,
                                   max_exact_lists=1, mc_samples=500, seed=7)
        self.assertEqual(first.estimate["method"], "monte_carlo")
        self.assertEqual(first.randomized_action_probs, second.randomized_action_probs)
        self.assertGreaterEqual(first.estimate["monte_carlo_standard_error_max"], 0)
        self.assertAlmostEqual(sum(first.randomized_action_probs.values()), 1.0, places=6)


class ItemVersionTests(unittest.TestCase):
    def test_as_of_ignores_later_known_version(self):
        rows = [
            {"item_key": "f", "category": "family",
             "effective_at": "2026-09-01T00:00:00+08:00",
             "known_at": "2026-09-01T00:00:00+08:00", "open": True},
            {"item_key": "f", "category": "family",
             "effective_at": "2026-10-20T00:00:00+08:00",
             "known_at": "2026-10-19T00:00:00+08:00", "open": False},
        ]
        catalog = load_catalog(rows)
        before = catalog.as_of("f", parse_time("2026-10-18T00:00:00+08:00", "t"))
        after = catalog.as_of("f", parse_time("2026-10-20T00:00:00+08:00", "t"))
        self.assertTrue(before.open)
        self.assertFalse(after.open)

    def test_effective_before_known_does_not_leak(self):
        rows = [{"item_key": "x", "category": "dining",
                 "effective_at": "2026-10-01T00:00:00+08:00",
                 "known_at": "2026-10-15T00:00:00+08:00", "open": False}]
        catalog = load_catalog(rows)
        with self.assertRaises(NoVersionError):
            catalog.as_of("x", parse_time("2026-10-05T00:00:00+08:00", "t"))


class BaselineTests(unittest.TestCase):
    def test_cold_start_uses_prior_not_zero(self):
        impressions, _ = deduplicate([impression()])
        baseline = train_popularity_baseline(impressions)
        self.assertAlmostEqual(baseline.score("new-item"), baseline.prior_mean)
        self.assertTrue(baseline.explain("new-item")["cold_start"])

    def test_baseline_never_inherits_future_clicks(self):
        train, _ = deduplicate([impression(reward_click=0)])
        baseline = train_popularity_baseline(train)
        self.assertTrue(0.0 < baseline.score("future-hit") < 1.0)
        self.assertEqual(baseline.clicks.get("future-hit", 0), 0)


class EndToEndTests(unittest.TestCase):
    def test_pipeline_and_determinism(self):
        with tempfile.TemporaryDirectory() as directory:
            first = run_pipeline(FIXTURES, Path(directory) / "a")
            second = run_pipeline(FIXTURES, None)
            self.assertTrue(first.payload["hand_check"]["assertions"]["ips_ok"])
            self.assertTrue(first.payload["time_leakage_audit"]["pass"])
            self.assertTrue(first.payload["test_evaluation"]["learned_ranker"]
                            ["unclipped"]["ips"] is not None)
            self.assertEqual(
                first.payload["ranker_comparison"]["predictive_weighted_log_loss"]["learned_ranker"],
                second.payload["ranker_comparison"]["predictive_weighted_log_loss"]["learned_ranker"],
            )
            self.assertTrue((Path(directory) / "a" / "report.json").is_file())

    def test_pipeline_does_not_mutate_fixtures(self):
        before = {p.name: p.read_bytes() for p in FIXTURES.iterdir() if p.is_file()}
        run_pipeline(FIXTURES, None)
        after = {p.name: p.read_bytes() for p in FIXTURES.iterdir() if p.is_file()}
        self.assertEqual(before, after)

    def test_self_check_passes(self):
        from src.selfcheck import run_self_check
        result = run_pipeline(FIXTURES, None)
        report = run_self_check(result.payload)
        self.assertTrue(report["passed"], [c for c in report["checks"] if not c["passed"]])


if __name__ == "__main__":
    unittest.main()
