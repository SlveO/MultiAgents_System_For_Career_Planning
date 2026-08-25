from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from project.experiments.architecture_protocol import ResearchContracts
from project.experiments.structured_output import (
    BACKEND_NAME,
    BACKEND_VERSION,
    INTEGRATION_NAME,
    LMFormatEnforcerBackend,
    StructuredOutputError,
    build_constraint_spec,
)


ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT.parents[1] / "models"
MODEL_DIRS = (
    MODELS / "Qwen3-VL-2B-Instruct",
    MODELS / "Qwen3-4B-Instruct-2507",
    MODELS / "Qwen3-VL-8B-Instruct",
)
HAS_LMFE = importlib.util.find_spec("lmformatenforcer") is not None
HAS_LOCAL_TOKENIZERS = all(
    (path / "tokenizer_config.json").is_file() for path in MODEL_DIRS
)
ARCHIVE = ROOT / "data/experiments/archive/pre-structured-output-v1"
HAS_FROZEN_BASELINE = all(
    (ARCHIVE / "manifests" / f"{name}.json").is_file()
    for name in ("v2", "v3")
)


def _evidence(evidence_id: str = "f-1", *, packet_type: str = "initial") -> dict:
    return {
        "schema_version": "research-evidence-v1",
        "packet_type": packet_type,
        "request_id": None if packet_type == "initial" else "r-1",
        "facts": [
            {
                "evidence_id": evidence_id,
                "fact": "图中写有 SQL 技能要求",
                "supporting_detail": "可直接读取中文和 SQL",
                "source_location": "图片左上角",
                "confidence": 0.9,
            }
        ],
        "missing_fields": [],
        "conflicts": [],
    }


def _plan(*, supports: object = None, evidence_ref: str = "f-1") -> dict:
    return {
        "schema_version": "research-plan-v1",
        "target_roles": ["数据分析师"],
        "gap_analysis": ["需要补齐项目证据"],
        "roadmap_30_90_180": [
            {
                "period": "30d",
                "objective": "学习",
                "deliverables": ["学习笔记"],
                "metrics": ["完成一次复盘"],
            },
            {
                "period": "90d",
                "objective": "项目",
                "deliverables": ["项目仓库"],
                "metrics": ["完成一次验收"],
            },
            {
                "period": "180d",
                "objective": "投递",
                "deliverables": ["投递复盘"],
                "metrics": ["完成一次迭代"],
            },
        ],
        "learning_resources": [],
        "next_actions": ["开始执行"],
        "risk_flags": [],
        "user_facing_advice": "按证据稳步执行。",
        "confidence": 0.6,
        "evidence_status": "sufficient",
        "missing_evidence": [],
        "evidence_used": [
            {
                "evidence_ref": evidence_ref,
                "fact": "SQL",
                "source_location": "图片左上角",
                "supports": ["target_roles"] if supports is None else supports,
            }
        ],
        "knowledge_ids_used": ["career-001"],
    }


def _lmfe_accepts(spec, payload: dict) -> bool:
    from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser

    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    parser = JsonSchemaParser(
        spec.decoding_schema,
        CharacterLevelParserConfig(
            alphabet="".join(chr(value) for value in range(0x10000))
        ),
    )
    try:
        for character in text:
            if character not in parser.get_allowed_characters():
                return False
            parser = parser.add_character(character)
    except Exception:
        return False
    return parser.can_end()


def _prefix_accepts(tokenizer, prefix, text: str) -> tuple[bool, list[int]]:
    import torch

    sequence = tokenizer.encode("约束测试", add_special_tokens=False)
    for token_id in tokenizer.encode(text, add_special_tokens=False):
        allowed = prefix(0, torch.tensor(sequence, dtype=torch.long))
        if token_id not in allowed:
            return False, allowed
        sequence.append(token_id)
    return True, prefix(0, torch.tensor(sequence, dtype=torch.long))


class TestProjectionWithoutHeavyDependencies(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contracts = ResearchContracts()

    def test_default_import_does_not_require_gpu_or_constraint_packages(self) -> None:
        code = (
            "import sys; "
            "sys.modules['torch']=None; sys.modules['transformers']=None; "
            "sys.modules['lmformatenforcer']=None; "
            "import project.experiments.structured_output; "
            "import project.experiments.run_architecture_experiments"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_projection_hashes_are_deterministic(self) -> None:
        first = build_constraint_spec(
            self.contracts,
            "plan",
            allowed_evidence_ids=["f-1", "f-2"],
            allowed_knowledge_ids=["career-001"],
        )
        second = build_constraint_spec(
            self.contracts,
            "plan",
            allowed_evidence_ids=["f-1", "f-2"],
            allowed_knowledge_ids=["career-001"],
        )
        self.assertEqual(first.decoding_schema, second.decoding_schema)
        self.assertEqual(first.decoding_schema_sha256, second.decoding_schema_sha256)
        self.assertEqual(
            first.canonical_schema_sha256,
            self.contracts.schema_sha256("plan"),
        )
        self.assertEqual(
            first.dynamic_enumeration_sources["evidence_ref"],
            "input_evidence.facts[*].evidence_id",
        )
        self.assertNotIn("expected_evidence", json.dumps(first.decoding_schema))

    def test_4b_and_8b_share_the_same_canonical_plan_schema(self) -> None:
        structured = build_constraint_spec(
            self.contracts,
            "plan",
            allowed_evidence_ids=["f-1"],
            allowed_knowledge_ids=["career-001"],
        )
        raw_media = build_constraint_spec(
            self.contracts,
            "plan",
            allowed_evidence_ids=None,
            allowed_knowledge_ids=["career-001"],
        )
        self.assertEqual(
            structured.canonical_schema_sha256,
            raw_media.canonical_schema_sha256,
        )
        self.assertNotEqual(
            structured.decoding_schema_sha256,
            raw_media.decoding_schema_sha256,
        )

    def test_invalid_dynamic_ids_fail_before_constraint_compilation(self) -> None:
        with self.assertRaises(StructuredOutputError):
            build_constraint_spec(
                self.contracts,
                "plan",
                allowed_evidence_ids=["f1"],
                allowed_knowledge_ids=["career-001"],
            )
        with self.assertRaises(StructuredOutputError):
            build_constraint_spec(
                self.contracts,
                "evidence",
                packet_type="delta",
                request_id="r1",
            )

    def test_missing_backend_fails_closed(self) -> None:
        with mock.patch(
            "project.experiments.structured_output.importlib.metadata.version",
            side_effect=importlib.metadata.PackageNotFoundError,
        ):
            with self.assertRaisesRegex(StructuredOutputError, "not installed"):
                LMFormatEnforcerBackend(object())


@unittest.skipUnless(
    HAS_FROZEN_BASELINE,
    "ignored v2/v3 evidence archive is available only on the L20 host",
)
class TestFrozenBaselineIntegrity(unittest.TestCase):
    def test_v2_and_v3_sources_still_match_frozen_manifests(self) -> None:
        for name in ("v2", "v3"):
            manifest_path = ARCHIVE / "manifests" / f"{name}.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for item in manifest["files"]:
                with self.subTest(version=name, path=item["path"]):
                    source = ROOT / item["path"]
                    self.assertTrue(source.is_file())
                    self.assertEqual(source.stat().st_size, item["size_bytes"])
                    self.assertEqual(
                        hashlib.sha256(source.read_bytes()).hexdigest(),
                        item["sha256"],
                    )


@unittest.skipUnless(HAS_LMFE, "lm-format-enforcer is optional outside the L20 profile")
class TestLMFEProjectionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contracts = ResearchContracts()

    def test_evidence_id_and_request_id_are_constrained(self) -> None:
        initial = build_constraint_spec(
            self.contracts, "evidence", packet_type="initial"
        )
        self.assertTrue(_lmfe_accepts(initial, _evidence("f-1")))
        self.assertFalse(_lmfe_accepts(initial, _evidence("f1")))

        delta = build_constraint_spec(
            self.contracts,
            "evidence",
            packet_type="delta",
            request_id="r-1",
        )
        self.assertTrue(
            _lmfe_accepts(delta, _evidence("f-1", packet_type="delta"))
        )
        wrong_request = _evidence("f-1", packet_type="delta")
        wrong_request["request_id"] = "r-2"
        self.assertFalse(_lmfe_accepts(delta, wrong_request))

    def test_supports_and_input_enums_are_constrained(self) -> None:
        spec = build_constraint_spec(
            self.contracts,
            "plan",
            allowed_evidence_ids=["f-1"],
            allowed_knowledge_ids=["career-001"],
        )
        self.assertTrue(_lmfe_accepts(spec, _plan()))
        self.assertFalse(_lmfe_accepts(spec, _plan(supports="target_roles")))
        self.assertFalse(_lmfe_accepts(spec, _plan(supports=["deliverables"])))
        self.assertFalse(_lmfe_accepts(spec, _plan(supports=["metrics"])))
        self.assertFalse(_lmfe_accepts(spec, _plan(evidence_ref="f-2")))
        wrong_knowledge = _plan()
        wrong_knowledge["knowledge_ids_used"] = ["career-002"]
        self.assertFalse(_lmfe_accepts(spec, wrong_knowledge))

    def test_initial_delta_plan_and_decision_compile(self) -> None:
        specs = (
            build_constraint_spec(
                self.contracts, "evidence", packet_type="initial"
            ),
            build_constraint_spec(
                self.contracts,
                "evidence",
                packet_type="delta",
                request_id="r-1",
            ),
            build_constraint_spec(
                self.contracts,
                "plan",
                allowed_evidence_ids=["f-1"],
                allowed_knowledge_ids=["career-001"],
            ),
            build_constraint_spec(self.contracts, "decision"),
        )
        from lmformatenforcer import JsonSchemaParser

        for spec in specs:
            with self.subTest(projection=spec.projection_type):
                parser = JsonSchemaParser(spec.decoding_schema)
                self.assertFalse(parser.can_end())


@unittest.skipUnless(
    HAS_LMFE and HAS_LOCAL_TOKENIZERS,
    "real local Qwen tokenizers are available only on the L20 host",
)
class TestQwenTokenizerCompatibility(unittest.TestCase):
    def test_transformers_5_and_all_three_tokenizers_filter_prefixes(self) -> None:
        from transformers import AutoTokenizer

        self.assertEqual(importlib.metadata.version(BACKEND_NAME), BACKEND_VERSION)
        self.assertEqual(importlib.metadata.version("transformers"), "5.15.1")
        contracts = ResearchContracts()
        spec = build_constraint_spec(
            contracts, "evidence", packet_type="initial"
        )
        valid_text = json.dumps(
            _evidence("f-1"), ensure_ascii=False, separators=(",", ":")
        )
        invalid_text = json.dumps(
            _evidence("f1"), ensure_ascii=False, separators=(",", ":")
        )
        fingerprints = set()
        for model_dir in MODEL_DIRS:
            with self.subTest(model=model_dir.name):
                tokenizer = AutoTokenizer.from_pretrained(
                    model_dir, local_files_only=True
                )
                chinese = '{"事实":"中文可往返","evidence_id":"f-1"}'
                self.assertEqual(
                    tokenizer.decode(
                        tokenizer.encode(chinese, add_special_tokens=False),
                        skip_special_tokens=False,
                    ),
                    chinese,
                )
                backend = LMFormatEnforcerBackend(tokenizer)
                prefix, telemetry = backend.prepare(spec)
                self.assertEqual(telemetry["integration"], INTEGRATION_NAME)
                self.assertEqual(
                    telemetry["canonical_schema_sha256"],
                    spec.canonical_schema_sha256,
                )
                self.assertEqual(
                    telemetry["decoding_schema_sha256"],
                    spec.decoding_schema_sha256,
                )
                self.assertFalse(telemetry["fallback_used"])
                self.assertFalse(telemetry["repair_used"])
                fingerprints.add(telemetry["tokenizer_fingerprint_sha256"])
                accepted, allowed = _prefix_accepts(
                    tokenizer, prefix, valid_text
                )
                self.assertTrue(accepted)
                self.assertIn(tokenizer.eos_token_id, allowed)

        self.assertEqual(len(fingerprints), 1)

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIRS[0], local_files_only=True
        )
        backend = LMFormatEnforcerBackend(tokenizer)
        invalid_prefix, _ = backend.prepare(spec)
        accepted, _ = _prefix_accepts(tokenizer, invalid_prefix, invalid_text)
        self.assertFalse(accepted)

        whitespace_prefix, _ = backend.prepare(spec)
        accepted, allowed = _prefix_accepts(tokenizer, whitespace_prefix, valid_text)
        self.assertTrue(accepted)
        sequence = tokenizer.encode("约束测试", add_special_tokens=False)
        sequence.extend(tokenizer.encode(valid_text, add_special_tokens=False))
        space_id = tokenizer.encode(" ", add_special_tokens=False)[0]
        import torch

        for _ in range(13):
            allowed = whitespace_prefix(
                0, torch.tensor(sequence, dtype=torch.long)
            )
            if space_id not in allowed:
                break
            sequence.append(space_id)
        self.assertNotIn(space_id, allowed)
        self.assertIn(tokenizer.eos_token_id, allowed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
