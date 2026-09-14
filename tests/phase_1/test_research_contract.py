from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "schemas" / "semantic_requirements.schema.json"
CONTRACT_PATH = ROOT / "configs" / "research_contract.json"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


class SemanticSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = load_json(SCHEMA_PATH)
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(cls.schema)
        cls.explicit = load_json(FIXTURES / "semantic_valid_explicit.json")
        cls.unknown = load_json(FIXTURES / "semantic_valid_unknown.json")

    def assertInvalid(self, instance: dict) -> None:  # noqa: N802 - unittest style
        with self.assertRaises(ValidationError):
            self.validator.validate(instance)

    def test_explicit_fixture_is_valid(self) -> None:
        self.validator.validate(self.explicit)

    def test_unknown_fixture_is_valid(self) -> None:
        self.validator.validate(self.unknown)

    def test_extra_property_is_rejected(self) -> None:
        candidate = copy.deepcopy(self.explicit)
        candidate["recommended_action"] = "edge_0"
        self.assertInvalid(candidate)

    def test_explicit_label_requires_evidence(self) -> None:
        candidate = copy.deepcopy(self.explicit)
        candidate["privacy"]["evidence"] = []
        self.assertInvalid(candidate)

    def test_not_stated_latency_cannot_contain_a_deadline(self) -> None:
        candidate = copy.deepcopy(self.unknown)
        candidate["latency"]["max_ms"] = 50
        self.assertInvalid(candidate)

    def test_not_stated_execution_policy_cannot_forbid_cloud(self) -> None:
        candidate = copy.deepcopy(self.unknown)
        candidate["execution_policy"]["cloud"] = "forbidden"
        self.assertInvalid(candidate)

    def test_out_of_range_confidence_is_rejected(self) -> None:
        candidate = copy.deepcopy(self.explicit)
        candidate["overall_confidence"] = 1.01
        self.assertInvalid(candidate)

    def test_schema_contains_no_action_output(self) -> None:
        serialized = json.dumps(self.schema).lower()
        self.assertNotIn("recommended_action", serialized)
        self.assertNotIn("oracle_action", serialized)


class ResearchContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_json(CONTRACT_PATH)

    def test_contract_and_documented_schema_versions_match(self) -> None:
        self.assertEqual(self.contract["contract_version"], "1.0.0")
        self.assertEqual(
            self.contract["semantic_isolation"]["llm_output_schema"],
            "schemas/semantic_requirements.schema.json",
        )
        self.assertTrue(SCHEMA_PATH.is_file())

    def test_llm_is_isolated_from_physics_and_answers(self) -> None:
        semantic = self.contract["semantic_isolation"]
        self.assertEqual(
            set(semantic["llm_inputs"]), {"task_text", "user_policy_text"}
        )
        required_forbidden = {
            "network_state",
            "server_load",
            "queue_state",
            "oracle_action",
            "reward",
            "outcome",
        }
        self.assertTrue(required_forbidden.issubset(semantic["llm_forbidden_inputs"]))
        self.assertFalse(semantic["live_llm_calls_during_rl"])
        self.assertFalse(semantic["reward_for_llm_agreement"])

    def test_initial_action_space_is_small_and_discrete(self) -> None:
        model = self.contract["decision_model"]
        self.assertEqual(
            model["action_space"],
            ["local", "edge_0", "edge_1", "edge_2", "cloud"],
        )
        self.assertTrue(model["initial_scope"]["binary_offloading_only"])
        self.assertFalse(model["initial_scope"]["partial_offloading"])
        self.assertFalse(model["initial_scope"]["resource_allocation_action"])

    def test_annotation_is_full_double_human_and_blind(self) -> None:
        annotation = self.contract["annotation"]
        self.assertEqual(annotation["independent_human_annotators"], 2)
        self.assertGreaterEqual(annotation["main_unique_items"], 240)
        self.assertTrue(annotation["both_annotate_every_main_item"])
        self.assertTrue(annotation["blind_independent_pass"])
        self.assertFalse(annotation["model_suggestions_visible"])
        self.assertIn("actual humans", annotation["annotator_claim"])

    def test_data_claim_is_hybrid_and_bupt_is_license_gated(self) -> None:
        data = self.contract["data_strategy"]
        self.assertEqual(data["benchmark_name"], "trace_driven_hybrid")
        self.assertFalse(data["fully_real_claim_allowed"])
        self.assertFalse(data["row_index_join_across_unrelated_sources"])
        self.assertFalse(data["raw_data_committed_to_git"])
        sources = {source["id"]: source for source in data["sources"]}
        self.assertEqual(
            sources["bupt_edge_computing_dataset"]["status"], "license_pending"
        )
        self.assertFalse(
            sources["bupt_edge_computing_dataset"]["raw_redistribution"]
        )
        self.assertEqual(
            sources["uci_mec_execution_times_859"]["status"],
            "license_verified_cc_by_4_0",
        )

    def test_reward_cannot_pay_for_llm_agreement(self) -> None:
        reward = self.contract["reward"]
        self.assertFalse(reward["llm_agreement_component"])
        self.assertNotIn("llm", reward["formula"].lower())
        self.assertEqual(reward["normalization_fit"], "training_split_only")
        self.assertFalse(reward["separate_queue_double_penalty"])

    def test_statistical_floor_and_collapse_alarm_are_predeclared(self) -> None:
        evaluation = self.contract["evaluation"]
        self.assertGreaterEqual(evaluation["training_seeds_minimum"], 5)
        self.assertGreaterEqual(
            evaluation["training_seeds_preferred"],
            evaluation["training_seeds_minimum"],
        )
        self.assertEqual(evaluation["confidence_interval"], 0.95)
        self.assertLessEqual(
            evaluation["collapse_alarm_single_action_fraction"], 0.9
        )
        self.assertTrue(evaluation["effect_size_required"])

    def test_testbed_is_not_required_for_core_claim(self) -> None:
        testbed = self.contract["testbed"]
        self.assertFalse(testbed["required_for_core_claim"])
        self.assertEqual(testbed["status"], "optional_extension")


if __name__ == "__main__":
    unittest.main()
