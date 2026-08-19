from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from project.core.schemas import CareerPlanResponse


class TestResearchContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[2]
        cls.dataset = root / "dataset"
        cls.schema_names = (
            "research_case.schema.json",
            "research_evidence.schema.json",
            "research_decision.schema.json",
            "research_plan.schema.json",
        )
        cls.schema_texts = {
            name: (cls.dataset / name).read_text(encoding="utf-8")
            for name in cls.schema_names
        }
        cls.schemas = {
            name: json.loads(text) for name, text in cls.schema_texts.items()
        }
        cls.protocol_text = (cls.dataset / "research_protocol.json").read_text(
            encoding="utf-8"
        )
        cls.protocol = json.loads(cls.protocol_text)
        cls.decision_log_text = (root / "docs" / "research-decisions.md").read_text(
            encoding="utf-8"
        )

    def test_case_contract_freezes_media_evidence_and_privacy_fields(self) -> None:
        case_schema = self.schemas["research_case.schema.json"]
        required = set(case_schema["required"])
        self.assertTrue(
            {
                "case_id",
                "modality",
                "asset_path",
                "asset_sha256",
                "source",
                "fixed_profile",
                "knowledge_snippets",
                "expected_evidence",
                "decisive_evidence_not_in_text",
                "privacy_reviewed",
            }.issubset(required)
        )
        evidence = case_schema["properties"]["expected_evidence"]
        self.assertEqual(evidence["minItems"], 3)
        self.assertEqual(evidence["maxItems"], 5)
        self.assertEqual(
            case_schema["properties"]["modality"]["enum"],
            ["image", "pdf_page"],
        )

    def test_protocol_references_each_versioned_contract(self) -> None:
        self.assertEqual(
            self.protocol["schema_refs"],
            {
                "case": "dataset/research_case.schema.json",
                "evidence": "dataset/research_evidence.schema.json",
                "decision": "dataset/research_decision.schema.json",
                "plan": "dataset/research_plan.schema.json",
            },
        )
        for relative_path in self.protocol["schema_refs"].values():
            self.assertTrue((self.dataset.parent / relative_path).is_file())

    def test_protocol_freezes_models_and_media_boundaries(self) -> None:
        groups = self.protocol["groups"]
        self.assertEqual(
            groups["monolithic"]["model_id"],
            "Qwen/Qwen3-VL-8B-Instruct",
        )
        self.assertTrue(groups["monolithic"]["planner_receives_raw_media"])
        self.assertIn("raw_media", groups["monolithic"]["planner_inputs"])

        for group_name in ["modular_one_shot", "modular_collaborative"]:
            group = groups[group_name]
            self.assertEqual(
                group["perceiver_model_id"],
                "Qwen/Qwen3-VL-2B-Instruct",
            )
            self.assertEqual(
                group["reasoner_model_id"],
                "Qwen/Qwen3-4B-Instruct-2507",
            )
            self.assertTrue(group["perceiver_receives_raw_media"])
            self.assertFalse(group["reasoner_receives_raw_media"])
            self.assertEqual(
                group["perceiver_inputs"],
                ["user_goal", "target_role", "raw_media", "evidence_schema"],
            )
            self.assertIn("fixed_profile", group["reasoner_inputs"])
            self.assertIn("knowledge_snippets", group["reasoner_inputs"])
            self.assertNotIn("raw_media", group["reasoner_inputs"])

        collaborative = groups["modular_collaborative"]
        self.assertEqual(collaborative["pilot_round_caps"], [2, 3])
        self.assertIsNone(collaborative["primary_round_cap"])

    def test_generation_and_runtime_settings_are_reproducible(self) -> None:
        generation = self.protocol["generation"]
        self.assertEqual(
            generation,
            {
                "decoding": "greedy",
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": 2048,
                "seed": 42,
                "reasoning_mode": "instruct_non_thinking",
            },
        )
        self.assertNotIn("temperature", generation)
        self.assertNotIn("top_p", generation)
        self.assertNotIn("thinking", generation)

        runtime = self.protocol["runtime"]
        self.assertEqual(runtime["dtype"], "bfloat16")
        self.assertTrue(runtime["eval_mode"])
        self.assertTrue(runtime["pin_model_revision"])
        self.assertTrue(runtime["revision_freeze_required_before_pilot"])
        self.assertEqual(runtime["pdf_render_dpi"], 144)
        self.assertEqual(runtime["visual_token_min"], 256)
        self.assertEqual(runtime["visual_token_max"], 1280)

    def test_prompt_inputs_and_collaboration_semantics_are_frozen(self) -> None:
        prompts = self.protocol["prompts"]
        perception_initial = prompts["perception"]["initial_user_template"]
        self.assertIn("{user_goal}", perception_initial)
        self.assertIn("{target_role}", perception_initial)
        self.assertNotIn("fixed_profile", perception_initial)
        self.assertNotIn("knowledge_snippets", perception_initial)

        planning = prompts["planning"]
        self.assertIn("{fixed_profile}", planning["common_user_template"])
        self.assertIn("{knowledge_snippets}", planning["common_user_template"])
        self.assertIn("raw_media", planning["raw_media_adapter"])
        self.assertIn(
            "structured_evidence", planning["structured_evidence_adapter"]
        )

        prompt_contract = self.protocol["prompt_contract"]
        self.assertEqual(prompt_contract["version"], "research-prompts-v2")
        self.assertEqual(
            set(prompt_contract["hidden_from_models"]),
            {"experiment_group", "expected_evidence", "scoring_notes"},
        )
        self.assertTrue(prompt_contract["strict_json"])
        self.assertTrue(prompt_contract["markdown_forbidden"])
        self.assertTrue(prompt_contract["chain_of_thought_forbidden"])

        collaboration = self.protocol["collaboration"]
        self.assertEqual(
            collaboration["round_definition"],
            "one_request_plus_one_perceiver_delta_response",
        )
        self.assertFalse(collaboration["decision_call_counts_as_round"])
        self.assertEqual(collaboration["clarification_response"], "delta_only")
        self.assertTrue(collaboration["preserve_conflicts"])
        self.assertIn("only_if_decisive_evidence_is_still_missing", collaboration["at_cap"])
        decision_template = prompts["collaboration_decision"]["user_template"]
        self.assertIn("reason_code=evidence_sufficient", decision_template)
        self.assertIn("reason_code=round_cap_reached", decision_template)

    def test_prompt_hashes_match_exact_templates(self) -> None:
        prompts = self.protocol["prompts"]
        hashing = self.protocol["prompt_contract"]["hashing"]
        separator = hashing["separator"]
        prompt_sets = {
            "perception_initial": [
                prompts["perception"]["system"],
                prompts["perception"]["initial_user_template"],
            ],
            "perception_clarification": [
                prompts["perception"]["system"],
                prompts["perception"]["clarification_user_template"],
            ],
            "planning_raw_media": [
                prompts["planning"]["system"],
                prompts["planning"]["common_user_template"],
                prompts["planning"]["raw_media_adapter"],
            ],
            "planning_structured_evidence": [
                prompts["planning"]["system"],
                prompts["planning"]["common_user_template"],
                prompts["planning"]["structured_evidence_adapter"],
            ],
            "collaboration_decision": [
                prompts["collaboration_decision"]["system"],
                prompts["collaboration_decision"]["user_template"],
            ],
        }
        expected = {
            name: hashlib.sha256(separator.join(parts).encode("utf-8")).hexdigest()
            for name, parts in prompt_sets.items()
        }
        self.assertEqual(hashing["algorithm"], "sha256")
        self.assertEqual(hashing["encoding"], "utf-8")
        self.assertEqual(hashing["hashes"], expected)

    def test_intermediate_contracts_freeze_delta_and_request_shapes(self) -> None:
        evidence = self.schemas["research_evidence.schema.json"]
        evidence_required = set(evidence["required"])
        self.assertTrue(
            {"packet_type", "request_id", "facts", "missing_fields", "conflicts"}
            .issubset(evidence_required)
        )
        fact_required = set(
            evidence["properties"]["facts"]["items"]["required"]
        )
        self.assertEqual(
            fact_required,
            {
                "evidence_id",
                "fact",
                "supporting_detail",
                "source_location",
                "confidence",
            },
        )

        decision = self.schemas["research_decision.schema.json"]
        actions = {
            branch["properties"]["action"]["const"] for branch in decision["oneOf"]
        }
        self.assertEqual(actions, {"final", "request_evidence"})
        request_branch = next(
            branch
            for branch in decision["oneOf"]
            if branch["properties"]["action"]["const"] == "request_evidence"
        )
        self.assertEqual(request_branch["properties"]["target"]["const"], "vision")
        self.assertEqual(
            set(request_branch["required"]),
            {
                "schema_version",
                "action",
                "reason_code",
                "request_id",
                "target",
                "question",
                "required_fields",
            },
        )

    def test_plan_contract_extends_canonical_response_without_metadata_leakage(
        self,
    ) -> None:
        plan = self.schemas["research_plan.schema.json"]
        required = set(plan["required"])
        canonical = {
            "target_roles",
            "gap_analysis",
            "roadmap_30_90_180",
            "learning_resources",
            "next_actions",
            "risk_flags",
            "user_facing_advice",
            "confidence",
        }
        self.assertTrue(canonical.issubset(CareerPlanResponse.model_fields))
        self.assertTrue(canonical.issubset(required))
        self.assertTrue(
            {
                "schema_version",
                "evidence_status",
                "missing_evidence",
                "evidence_used",
                "knowledge_ids_used",
            }.issubset(required)
        )
        evidence_required = set(
            plan["properties"]["evidence_used"]["items"]["required"]
        )
        self.assertEqual(
            evidence_required,
            {"evidence_ref", "fact", "source_location", "supports"},
        )
        forbidden_metadata = {
            "case_id",
            "experiment_group",
            "model_id",
            "model_revision",
            "latency_ms",
            "peak_vram_mb",
            "retry_count",
            "session_id",
        }
        self.assertTrue(forbidden_metadata.isdisjoint(plan["properties"]))

    def test_formal_failure_and_evaluation_policies_match_confirmed_decisions(
        self,
    ) -> None:
        run_policy = self.protocol["formal_run_policy"]
        self.assertEqual(run_policy["response_parser"], "json.loads_then_json_schema")
        self.assertFalse(run_policy["strip_markdown_fences"])
        self.assertFalse(run_policy["automatic_json_repair"])
        self.assertEqual(run_policy["format_retry_count"], 0)
        self.assertEqual(run_policy["invalid_output_effective_quality_score"], 1)
        self.assertTrue(run_policy["run_metadata_outside_model_output"])

        evaluation = self.protocol["evaluation"]
        self.assertEqual(evaluation["blind_raters"], 2)
        self.assertEqual(len(evaluation["dimensions"]), 5)
        self.assertEqual(
            evaluation["adjudicate_when_dimension_difference_greater_than"], 1
        )
        selection = evaluation["pilot"]["select_three_only_if_all"]
        self.assertEqual(selection["minimum_evidence_mean_gain"], 0.5)
        self.assertEqual(selection["minimum_five_dimension_mean_gain"], 0.25)
        self.assertEqual(selection["minimum_cases_improved"], 3)
        self.assertEqual(selection["maximum_single_case_evidence_drop"], 1.0)
        self.assertTrue(evaluation["pilot"]["exclude_from_primary_comparison"])
        self.assertEqual(evaluation["primary_comparison"]["total_runs"], 60)

    def test_confirmed_decisions_are_archived(self) -> None:
        self.assertIn("# Research Decision Log", self.decision_log_text)
        for decision_id in range(1, 8):
            self.assertIn(f"RD-{decision_id:03d}", self.decision_log_text)
        self.assertEqual(self.decision_log_text.count("**Status:** Confirmed"), 7)
        self.assertIn("Superseded", self.decision_log_text)

    def test_contracts_contain_no_private_windows_paths_or_keys(self) -> None:
        combined = "".join(self.schema_texts.values())
        combined += self.protocol_text + self.decision_log_text
        self.assertNotIn(":\\Users\\", combined)
        self.assertNotIn("DEEPSEEK_API_KEY=", combined)


if __name__ == "__main__":
    unittest.main(verbosity=2)
