"""Lazy L20 model clients and real architecture adapters.

Heavy GPU and media dependencies are imported only inside runtime methods.
This module never downloads model weights.
"""
from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from project.experiments.architecture_protocol import (
    ACTION_REQUEST_EVIDENCE,
    ALL_GROUPS,
    BaseArchitectureAdapter,
    ContractValidationError,
    GROUP_MODULAR_COLLABORATIVE,
    GROUP_MODULAR_ONE_SHOT,
    GROUP_MONOLITHIC,
    ResearchContracts,
    group_model_ids,
)
from project.experiments.model_registry import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODELS_DIR,
    DEFAULT_PREFLIGHT_RECORD,
    DEFAULT_SEED,
    FROZEN_MODELS,
    ManualPreflightError,
    ModelNotDownloadedError,
    REPO_ROOT,
    get_entry,
    local_model_dir,
    validate_local_model,
    validate_manual_preflight,
    validate_preflight_device,
)
from project.experiments.structured_output import (
    LMFormatEnforcerBackend,
    StructuredOutputError,
    StructuredOutputSpec,
    build_constraint_spec,
)

DEFAULT_CACHE_DIR = REPO_ROOT / "data/experiments/cache"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
PDF_DPI = 144
PROMPT_SEPARATOR = "\n---\n"


class ModelOutputError(ValueError):
    """A model returned invalid JSON or violated a frozen schema."""

    def __init__(
        self, code: str, raw_output: Optional[str], detail: str
    ) -> None:
        self.code = code
        self.raw_output = raw_output
        self.detail = detail
        super().__init__(detail)


def _sha256_text(parts: Iterable[str]) -> str:
    return hashlib.sha256(PROMPT_SEPARATOR.join(parts).encode("utf-8")).hexdigest()


class PromptCatalog:
    """Render prompts only after verifying their frozen source hashes."""

    def __init__(self, contracts: ResearchContracts) -> None:
        self.protocol = contracts.protocol
        self.prompts = self.protocol["prompts"]
        contract = self.protocol["prompt_contract"]
        self.version = contract["version"]
        self.hashes = dict(contract["hashing"]["hashes"])
        self.schemas = {
            kind: self._json(
                json.loads(
                    (contracts.repo_root / relative_path).read_text(encoding="utf-8")
                )
            )
            for kind, relative_path in self.protocol["schema_refs"].items()
        }
        sections = {
            "perception_initial": [
                self.prompts["perception"]["system"],
                self.prompts["perception"]["initial_user_template"],
            ],
            "perception_clarification": [
                self.prompts["perception"]["system"],
                self.prompts["perception"]["clarification_user_template"],
            ],
            "planning_raw_media": [
                self.prompts["planning"]["system"],
                self.prompts["planning"]["common_user_template"],
                self.prompts["planning"]["raw_media_adapter"],
            ],
            "planning_structured_evidence": [
                self.prompts["planning"]["system"],
                self.prompts["planning"]["common_user_template"],
                self.prompts["planning"]["structured_evidence_adapter"],
            ],
            "collaboration_decision": [
                self.prompts["collaboration_decision"]["system"],
                self.prompts["collaboration_decision"]["user_template"],
            ],
        }
        for name, parts in sections.items():
            if _sha256_text(parts) != self.hashes[name]:
                raise ValueError(f"frozen prompt hash mismatch: {name}")

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def perception_initial(self, case: Mapping[str, Any]) -> Tuple[str, str, str]:
        entry = self.prompts["perception"]
        user = entry["initial_user_template"].format(
            user_goal=case["user_goal"], target_role=case["fixed_profile"]["target_role"]
        )
        return (
            entry["system"],
            f"{user}\noutput_schema={self.schemas['evidence']}",
            self.hashes["perception_initial"],
        )

    def perception_delta(
        self,
        case: Mapping[str, Any],
        request: Mapping[str, Any],
        evidence: Mapping[str, Any],
    ) -> Tuple[str, str, str]:
        entry = self.prompts["perception"]
        user = entry["clarification_user_template"].format(
            user_goal=case["user_goal"],
            target_role=case["fixed_profile"]["target_role"],
            request_evidence=self._json(request),
            current_evidence=self._json(evidence),
        )
        return (
            entry["system"],
            f"{user}\noutput_schema={self.schemas['evidence']}",
            self.hashes["perception_clarification"],
        )

    def planning(
        self, case: Mapping[str, Any], payload: Any, *, raw_media: bool
    ) -> Tuple[str, str, str]:
        entry = self.prompts["planning"]
        common = entry["common_user_template"].format(
            user_goal=case["user_goal"],
            fixed_profile=self._json(case["fixed_profile"]),
            knowledge_snippets=self._json(case["knowledge_snippets"]),
            input_payload="[RAW_MEDIA_ATTACHED]" if raw_media else self._json(payload),
        )
        adapter_name = (
            "raw_media_adapter" if raw_media else "structured_evidence_adapter"
        )
        hash_name = "planning_raw_media" if raw_media else "planning_structured_evidence"
        user = f"{common}\n{entry[adapter_name]}\noutput_schema={self.schemas['plan']}"
        return entry["system"], user, self.hashes[hash_name]

    def decision(
        self,
        case: Mapping[str, Any],
        evidence: Mapping[str, Any],
        rounds_remaining: int,
    ) -> Tuple[str, str, str]:
        entry = self.prompts["collaboration_decision"]
        user = entry["user_template"].format(
            user_goal=case["user_goal"],
            target_role=case["fixed_profile"]["target_role"],
            current_evidence=self._json(evidence),
            rounds_remaining=rounds_remaining,
        )
        return (
            entry["system"],
            f"{user}\noutput_schema={self.schemas['decision']}",
            self.hashes["collaboration_decision"],
        )


class LocalModelClient:
    """Load and generate from one already-downloaded frozen model."""

    def __init__(
        self,
        model_id: str,
        *,
        models_dir: Optional[Path] = None,
        seed: int = DEFAULT_SEED,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        device: str = "cuda:0",
    ) -> None:
        self.entry = get_entry(model_id)
        self.models_dir = Path(models_dir or DEFAULT_MODELS_DIR)
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.device = device
        self._model: Any = None
        self._processor: Any = None
        self._structured_output: Optional[LMFormatEnforcerBackend] = None

    def load(self) -> "LocalModelClient":
        target = validate_local_model(self.entry["model_id"], self.models_dir)
        import torch
        import transformers

        if not self.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("formal L20 inference requires an available CUDA device")
        try:
            with torch.cuda.device(self.device):
                bf16_supported = torch.cuda.is_bf16_supported()
        except (AssertionError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                f"invalid or unavailable CUDA device: {self.device}"
            ) from exc
        if not bf16_supported:
            raise RuntimeError("selected CUDA device does not support bfloat16")
        if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
            raise RuntimeError("installed Transformers lacks Qwen3-VL support")
        transformers.set_seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        if self.entry["kind"] == "vl":
            self._processor = transformers.AutoProcessor.from_pretrained(
                str(target),
                local_files_only=True,
                min_pixels=MIN_PIXELS,
                max_pixels=MAX_PIXELS,
            )
            model_class = transformers.Qwen3VLForConditionalGeneration
        else:
            self._processor = transformers.AutoTokenizer.from_pretrained(
                str(target), local_files_only=True
            )
            model_class = transformers.AutoModelForCausalLM
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        self._structured_output = LMFormatEnforcerBackend(tokenizer)
        self._model = model_class.from_pretrained(
            str(target),
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map={"": self.device},
        )
        self._model.eval()
        return self

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        media_path: Optional[Path] = None,
        constraint: StructuredOutputSpec,
    ) -> Dict[str, Any]:
        if (
            self._model is None
            or self._processor is None
            or self._structured_output is None
        ):
            raise RuntimeError("call load() before generate()")
        if not isinstance(constraint, StructuredOutputSpec):
            raise StructuredOutputError(
                "formal generation requires a structured-output constraint"
            )
        if self.entry["kind"] == "text" and media_path is not None:
            raise ValueError("text reasoner must never receive raw media")
        if self.entry["kind"] == "vl" and media_path is None:
            raise ValueError("vision-language model requires versioned media")
        import torch

        content: List[Dict[str, Any]] = []
        if media_path is not None:
            path = Path(media_path)
            if not path.is_file():
                raise FileNotFoundError("versioned media is missing")
            content.append({"type": "image", "url": str(path.resolve())})
        content.append({"type": "text", "text": user_prompt})
        if self.entry["kind"] == "vl":
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(self.device)
        visual_tokens = 0
        grid = inputs.get("image_grid_thw")
        if grid is not None:
            merge_size = int(
                getattr(self._processor.image_processor, "merge_size", 1)
            )
            visual_tokens = int(grid.prod(dim=1).sum().item() / (merge_size**2))
        prefix_allowed_tokens_fn, constraint_telemetry = (
            self._structured_output.prepare(constraint)
        )
        torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.max_new_tokens,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated)
        ]
        output_text = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return {
            "model_id": self.entry["model_id"],
            "revision": self.entry["revision"],
            "output_text": output_text,
            "latency_ms": latency_ms,
            "device": self.device,
            "peak_vram_mb": round(
                torch.cuda.max_memory_allocated(self.device) / 2**20
            ),
            "visual_tokens": visual_tokens,
            "seed": self.seed,
            "max_new_tokens": self.max_new_tokens,
            "structured_output": constraint_telemetry,
        }

    def close(self) -> None:
        had_model = self._model is not None
        self._model = None
        self._processor = None
        self._structured_output = None
        gc.collect()
        if had_model:
            import torch

            if torch.cuda.is_available():
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()


def _strict_payload(
    result: Mapping[str, Any], kind: str, contracts: ResearchContracts
) -> Dict[str, Any]:
    raw = str(result["output_text"])
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModelOutputError("invalid_json", raw, str(exc)) from exc
    if not isinstance(payload, dict):
        raise ModelOutputError("invalid_json_type", raw, "model JSON must be an object")
    try:
        contracts.validate(kind, payload)
    except ContractValidationError as exc:
        raise ModelOutputError("schema_invalid", raw, str(exc)) from exc
    return payload


def _execution_error(exc: BaseException) -> ModelOutputError:
    return ModelOutputError(
        "inference_failed",
        None,
        f"local inference failed with {type(exc).__name__}",
    )


def _constraint_error(exc: StructuredOutputError) -> ModelOutputError:
    return ModelOutputError(
        "constraint_failed",
        None,
        f"structured-output enforcement failed with {type(exc).__name__}",
    )


def _failure_plan(case: Mapping[str, Any], error_code: str) -> Dict[str, Any]:
    target = case["fixed_profile"]["target_role"]
    return {
        "schema_version": "research-plan-v1",
        "target_roles": [target],
        "gap_analysis": ["模型输出未通过冻结 JSON Schema，无法形成可评估规划"],
        "roadmap_30_90_180": [
            {
                "period": "30d",
                "objective": "暂停规划",
                "deliverables": ["保留失败记录"],
                "metrics": ["记录完整"],
            },
            {
                "period": "90d",
                "objective": "暂停规划",
                "deliverables": ["完成人工复核"],
                "metrics": ["复核一次"],
            },
            {
                "period": "180d",
                "objective": "暂停规划",
                "deliverables": ["等待合规重跑决策"],
                "metrics": ["不自动重试"],
            },
        ],
        "learning_resources": [],
        "next_actions": ["检查原始无效输出和环境记录"],
        "risk_flags": [f"形式失败：{error_code}"],
        "user_facing_advice": "本次输出无效，不应据此采取职业决策。",
        "confidence": 0.0,
        "evidence_status": "insufficient",
        "missing_evidence": ["缺少 Schema-valid 模型规划"],
        "evidence_used": [],
        "knowledge_ids_used": [],
    }


def _case_media(case: Mapping[str, Any], cache_dir: Path) -> Path:
    source = (REPO_ROOT / str(case["asset_path"])).resolve()
    source.relative_to((REPO_ROOT / "dataset/research_cases/assets").resolve())
    if case["modality"] == "image":
        return source
    if case["modality"] != "pdf_page":
        raise ValueError("unsupported research modality")
    import fitz

    page_index = int(case["page_number"]) - 1
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{case['case_id']}-page-{page_index + 1}-144dpi.png"
    with fitz.open(source) as document:
        if page_index < 0 or page_index >= document.page_count:
            raise ValueError("requested PDF page is outside the document")
        matrix = fitz.Matrix(PDF_DPI / 72, PDF_DPI / 72)
        document.load_page(page_index).get_pixmap(matrix=matrix).save(target)
    return target


def _evidence_refs_are_valid(
    plan: Mapping[str, Any],
    evidence: Mapping[str, Any],
    raw_output: Optional[str] = None,
) -> None:
    allowed = {item["evidence_id"] for item in evidence["facts"]}
    used = {item["evidence_ref"] for item in plan["evidence_used"]}
    if not used.issubset(allowed):
        raise ModelOutputError(
            "schema_invalid",
            raw_output or json.dumps(plan, ensure_ascii=False),
            "plan cited evidence_id absent from evidence packet",
        )


class _L20Adapter(BaseArchitectureAdapter):
    backend = "l20"

    def __init__(
        self,
        *,
        contracts: ResearchContracts,
        clients: Mapping[str, LocalModelClient],
        cache_dir: Path = DEFAULT_CACHE_DIR,
    ) -> None:
        self.contracts = contracts
        self.protocol = contracts.protocol
        self.catalog = PromptCatalog(contracts)
        self.clients = dict(clients)
        self.cache_dir = Path(cache_dir)
        self.model_ids = group_model_ids(self.protocol, self.group)

    def _constraint(
        self,
        kind: str,
        *,
        packet_type: Optional[str] = None,
        request_id: Optional[str] = None,
        evidence: Optional[Mapping[str, Any]] = None,
        case: Optional[Mapping[str, Any]] = None,
        raw_media: bool = False,
    ) -> StructuredOutputSpec:
        evidence_ids: Optional[List[str]] = None
        if kind == "plan" and not raw_media:
            evidence_ids = [
                str(item["evidence_id"])
                for item in (evidence or {}).get("facts", [])
            ]
        knowledge_ids = [
            str(item["knowledge_id"])
            for item in (case or {}).get("knowledge_snippets", [])
        ]
        return build_constraint_spec(
            self.contracts,
            kind,
            packet_type=packet_type,
            request_id=request_id,
            allowed_evidence_ids=evidence_ids,
            allowed_knowledge_ids=knowledge_ids,
        )

    def _record(
        self,
        case: Mapping[str, Any],
        *,
        round_cap: int,
        rounds_used: int,
        plan: Dict[str, Any],
        telemetry: Sequence[Mapping[str, Any]],
        prompt_hashes: Sequence[str],
        error: Optional[str] = None,
        raw_invalid_output: Optional[str] = None,
    ) -> Dict[str, Any]:
        peaks = [
            int(item["peak_vram_mb"])
            for item in telemetry
            if item.get("peak_vram_mb") is not None
        ]
        return {
            "case_id": case["case_id"],
            "group": self.group,
            "modality": case["modality"],
            "backend": self.backend,
            "round_cap": round_cap,
            "rounds_used": rounds_used,
            "finalized": True,
            "error": error,
            "model_ids": dict(self.model_ids),
            "model_revisions": {
                role: get_entry(model_id)["revision"]
                for role, model_id in self.model_ids.items()
            },
            "generation": dict(self.protocol["generation"]),
            "latency_ms": round(
                sum(float(item["latency_ms"]) for item in telemetry), 3
            ),
            "peak_vram_mb": max(peaks) if peaks else None,
            "device": sorted({str(item["device"]) for item in telemetry}),
            "visual_tokens": sum(
                int(item.get("visual_tokens", 0)) for item in telemetry
            ),
            "prompt_contract_version": self.catalog.version,
            "prompt_hashes": list(prompt_hashes),
            "retry_count": 0,
            "repair_used": False,
            "fallback_used": False,
            "pipeline_success": error is None,
            "semantic_content_valid": None,
            "evidence_faithfulness": None,
            "task_success": None,
            "structured_output": [
                dict(item["structured_output"])
                for item in telemetry
                if item.get("structured_output") is not None
            ],
            "raw_invalid_output": raw_invalid_output,
            "plan": plan,
        }

    def _failed(
        self,
        case: Mapping[str, Any],
        *,
        round_cap: int,
        rounds_used: int,
        telemetry: Sequence[Mapping[str, Any]],
        prompt_hashes: Sequence[str],
        exc: ModelOutputError,
    ) -> Dict[str, Any]:
        plan = _failure_plan(case, exc.code)
        self.contracts.validate("plan", plan)
        return self._record(
            case,
            round_cap=round_cap,
            rounds_used=rounds_used,
            plan=plan,
            telemetry=telemetry,
            prompt_hashes=prompt_hashes,
            error=exc.code,
            raw_invalid_output=exc.raw_output,
        )


class L20MonolithicAdapter(_L20Adapter):
    group = GROUP_MONOLITHIC

    def run(self, case: Dict[str, Any], round_cap: int = 2) -> Dict[str, Any]:
        telemetry: List[Mapping[str, Any]] = []
        prompt_hashes: List[str] = []
        try:
            media = _case_media(case, self.cache_dir)
            system, user, prompt_hash = self.catalog.planning(
                case, None, raw_media=True
            )
            prompt_hashes.append(prompt_hash)
            result = self.clients["planner"].generate(
                system,
                user,
                media_path=media,
                constraint=self._constraint(
                    "plan", case=case, raw_media=True
                ),
            )
            telemetry.append(result)
            plan = _strict_payload(result, "plan", self.contracts)
        except ModelOutputError as exc:
            return self._failed(
                case,
                round_cap=0,
                rounds_used=0,
                telemetry=telemetry,
                prompt_hashes=prompt_hashes,
                exc=exc,
            )
        except StructuredOutputError as exc:
            return self._failed(
                case,
                round_cap=0,
                rounds_used=0,
                telemetry=telemetry,
                prompt_hashes=prompt_hashes,
                exc=_constraint_error(exc),
            )
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            return self._failed(
                case,
                round_cap=0,
                rounds_used=0,
                telemetry=telemetry,
                prompt_hashes=prompt_hashes,
                exc=_execution_error(exc),
            )
        return self._record(
            case,
            round_cap=0,
            rounds_used=0,
            plan=plan,
            telemetry=telemetry,
            prompt_hashes=prompt_hashes,
        )


class L20ModularOneShotAdapter(_L20Adapter):
    group = GROUP_MODULAR_ONE_SHOT

    def run(self, case: Dict[str, Any], round_cap: int = 2) -> Dict[str, Any]:
        prompt_hashes: List[str] = []
        telemetry: List[Mapping[str, Any]] = []
        try:
            media = _case_media(case, self.cache_dir)
            system, user, prompt_hash = self.catalog.perception_initial(case)
            prompt_hashes.append(prompt_hash)
            perception = self.clients["perceiver"].generate(
                system,
                user,
                media_path=media,
                constraint=self._constraint(
                    "evidence", packet_type="initial"
                ),
            )
            telemetry.append(perception)
            evidence = _strict_payload(perception, "evidence", self.contracts)
            system, user, prompt_hash = self.catalog.planning(
                case, evidence, raw_media=False
            )
            prompt_hashes.append(prompt_hash)
            reasoning = self.clients["reasoner"].generate(
                system,
                user,
                constraint=self._constraint(
                    "plan", evidence=evidence, case=case
                ),
            )
            telemetry.append(reasoning)
            plan = _strict_payload(reasoning, "plan", self.contracts)
            _evidence_refs_are_valid(
                plan, evidence, str(reasoning["output_text"])
            )
        except ModelOutputError as exc:
            return self._failed(
                case,
                round_cap=0,
                rounds_used=0,
                telemetry=telemetry,
                prompt_hashes=prompt_hashes,
                exc=exc,
            )
        except StructuredOutputError as exc:
            return self._failed(
                case,
                round_cap=0,
                rounds_used=0,
                telemetry=telemetry,
                prompt_hashes=prompt_hashes,
                exc=_constraint_error(exc),
            )
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            return self._failed(
                case,
                round_cap=0,
                rounds_used=0,
                telemetry=telemetry,
                prompt_hashes=prompt_hashes,
                exc=_execution_error(exc),
            )
        return self._record(
            case,
            round_cap=0,
            rounds_used=0,
            plan=plan,
            telemetry=telemetry,
            prompt_hashes=prompt_hashes,
        )


class L20ModularCollaborativeAdapter(_L20Adapter):
    group = GROUP_MODULAR_COLLABORATIVE

    def run(self, case: Dict[str, Any], round_cap: int = 2) -> Dict[str, Any]:
        prompt_hashes: List[str] = []
        telemetry: List[Mapping[str, Any]] = []
        rounds_used = 0
        request_ids: set[str] = set()
        try:
            media = _case_media(case, self.cache_dir)
            system, user, prompt_hash = self.catalog.perception_initial(case)
            prompt_hashes.append(prompt_hash)
            result = self.clients["perceiver"].generate(
                system,
                user,
                media_path=media,
                constraint=self._constraint(
                    "evidence", packet_type="initial"
                ),
            )
            telemetry.append(result)
            evidence = _strict_payload(result, "evidence", self.contracts)
            while True:
                remaining = round_cap - rounds_used
                system, user, prompt_hash = self.catalog.decision(
                    case, evidence, remaining
                )
                prompt_hashes.append(prompt_hash)
                decision_result = self.clients["reasoner"].generate(
                    system,
                    user,
                    constraint=self._constraint("decision"),
                )
                telemetry.append(decision_result)
                decision = _strict_payload(
                    decision_result, "decision", self.contracts
                )
                if decision["action"] != ACTION_REQUEST_EVIDENCE or remaining <= 0:
                    break
                if decision["request_id"] in request_ids:
                    raise ModelOutputError(
                        "schema_invalid",
                        str(decision_result["output_text"]),
                        "collaboration repeated request_id",
                    )
                request_ids.add(decision["request_id"])
                system, user, prompt_hash = self.catalog.perception_delta(
                    case, decision, evidence
                )
                prompt_hashes.append(prompt_hash)
                delta_result = self.clients["perceiver"].generate(
                    system,
                    user,
                    media_path=media,
                    constraint=self._constraint(
                        "evidence",
                        packet_type="delta",
                        request_id=str(decision["request_id"]),
                    ),
                )
                telemetry.append(delta_result)
                delta = _strict_payload(delta_result, "evidence", self.contracts)
                if (
                    delta["packet_type"] != "delta"
                    or delta["request_id"] != decision["request_id"]
                ):
                    raise ModelOutputError(
                        "schema_invalid",
                        str(delta_result["output_text"]),
                        "clarification response does not match request",
                    )
                facts = list(evidence["facts"]) + list(delta["facts"])
                identifiers = [item["evidence_id"] for item in facts]
                if len(identifiers) != len(set(identifiers)):
                    raise ModelOutputError(
                        "schema_invalid",
                        str(delta_result["output_text"]),
                        "clarification repeated evidence_id",
                    )
                evidence = {
                    **evidence,
                    "facts": facts,
                    "missing_fields": list(delta["missing_fields"]),
                    "conflicts": list(evidence["conflicts"])
                    + list(delta["conflicts"]),
                }
                rounds_used += 1
            system, user, prompt_hash = self.catalog.planning(
                case, evidence, raw_media=False
            )
            prompt_hashes.append(prompt_hash)
            plan_result = self.clients["reasoner"].generate(
                system,
                user,
                constraint=self._constraint(
                    "plan", evidence=evidence, case=case
                ),
            )
            telemetry.append(plan_result)
            plan = _strict_payload(plan_result, "plan", self.contracts)
            _evidence_refs_are_valid(
                plan, evidence, str(plan_result["output_text"])
            )
        except ModelOutputError as exc:
            return self._failed(
                case,
                round_cap=round_cap,
                rounds_used=rounds_used,
                telemetry=telemetry,
                prompt_hashes=prompt_hashes,
                exc=exc,
            )
        except StructuredOutputError as exc:
            return self._failed(
                case,
                round_cap=round_cap,
                rounds_used=rounds_used,
                telemetry=telemetry,
                prompt_hashes=prompt_hashes,
                exc=_constraint_error(exc),
            )
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            return self._failed(
                case,
                round_cap=round_cap,
                rounds_used=rounds_used,
                telemetry=telemetry,
                prompt_hashes=prompt_hashes,
                exc=_execution_error(exc),
            )
        return self._record(
            case,
            round_cap=round_cap,
            rounds_used=rounds_used,
            plan=plan,
            telemetry=telemetry,
            prompt_hashes=prompt_hashes,
        )


def build_l20_adapters(
    contracts: ResearchContracts,
    models_dir: Path = DEFAULT_MODELS_DIR,
    device: str = "cuda:0",
    *,
    preflight_record: Path = DEFAULT_PREFLIGHT_RECORD,
    groups: Optional[Sequence[str]] = None,
) -> Dict[str, BaseArchitectureAdapter]:
    """Build only the requested real adapters after a fresh human PASS record."""
    selected_groups = list(groups or ALL_GROUPS)
    if not selected_groups or any(
        group not in ALL_GROUPS for group in selected_groups
    ):
        raise ValueError("groups must contain known architecture groups")
    preflight = validate_manual_preflight(preflight_record)
    validate_preflight_device(preflight, device)
    generation = contracts.protocol["generation"]
    required_roles = set()
    if GROUP_MONOLITHIC in selected_groups:
        required_roles.add("monolithic")
    if any(
        group in selected_groups
        for group in (GROUP_MODULAR_ONE_SHOT, GROUP_MODULAR_COLLABORATIVE)
    ):
        required_roles.update(("perceiver", "reasoner"))
    clients = {
        entry["role"]: LocalModelClient(
            entry["model_id"],
            models_dir=Path(models_dir),
            device=device,
            seed=int(generation["seed"]),
            max_new_tokens=int(generation["max_new_tokens"]),
        )
        for entry in FROZEN_MODELS
        if entry["role"] in required_roles
    }
    loaded: List[LocalModelClient] = []
    try:
        for client in clients.values():
            client.load()
            loaded.append(client)
    except Exception:
        for client in loaded:
            client.close()
        raise
    adapters: Dict[str, BaseArchitectureAdapter] = {}
    if GROUP_MONOLITHIC in selected_groups:
        adapters[GROUP_MONOLITHIC] = L20MonolithicAdapter(
            contracts=contracts, clients={"planner": clients["monolithic"]}
        )
    modular = {
        role: clients[role]
        for role in ("perceiver", "reasoner")
        if role in clients
    }
    if GROUP_MODULAR_ONE_SHOT in selected_groups:
        adapters[GROUP_MODULAR_ONE_SHOT] = L20ModularOneShotAdapter(
            contracts=contracts, clients=modular
        )
    if GROUP_MODULAR_COLLABORATIVE in selected_groups:
        adapters[GROUP_MODULAR_COLLABORATIVE] = L20ModularCollaborativeAdapter(
            contracts=contracts, clients=modular
        )
    return adapters


def close_adapters(adapters: Mapping[str, BaseArchitectureAdapter]) -> None:
    seen: set[int] = set()
    for adapter in adapters.values():
        for client in getattr(adapter, "clients", {}).values():
            if id(client) not in seen:
                seen.add(id(client))
                client.close()


def _validate_smoke_payload(
    entry: Mapping[str, str], payload: Mapping[str, Any], raw_output: str
) -> None:
    if entry["role"] == "monolithic" and not payload["evidence_used"]:
        raise ModelOutputError(
            "smoke_grounding_failed",
            raw_output,
            "monolithic smoke plan cited no fact observed in the image",
        )
    if entry["role"] == "perceiver":
        text = json.dumps(payload["facts"], ensure_ascii=False).casefold()
        if not any(keyword in text for keyword in ("sql", "仪表盘", "dashboard")):
            raise ModelOutputError(
                "smoke_grounding_failed",
                raw_output,
                "perceiver smoke found neither the SQL nor dashboard fixture fact",
            )


def _smoke_outcome_fields(
    error: Optional[str], telemetry: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    inference_success = telemetry is not None
    raw_json_valid: Optional[bool]
    schema_valid: Optional[bool]
    if not inference_success:
        raw_json_valid = None
        schema_valid = None
    else:
        raw_json_valid = error != "invalid_json"
        schema_valid = error not in {
            "invalid_json",
            "invalid_json_type",
            "schema_invalid",
        }
    return {
        "inference_success": inference_success,
        "raw_json_valid": raw_json_valid,
        "schema_valid": schema_valid,
        "repair_used": False,
        "fallback_used": False,
        "pipeline_success": bool(
            inference_success and raw_json_valid and schema_valid
        ),
        "semantic_content_valid": None,
        "evidence_faithfulness": None,
        "task_success": None,
    }


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run media-grounded smoke checks for all three L20 models."
    )
    parser.add_argument(
        "--model",
        choices=["all"] + [entry["model_id"] for entry in FROZEN_MODELS],
        default="all",
    )
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--preflight-record", default=str(DEFAULT_PREFLIGHT_RECORD))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "data/experiments/smoke"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    preflight = validate_manual_preflight(Path(args.preflight_record))
    validate_preflight_device(preflight, args.device)
    contracts = ResearchContracts()
    catalog = PromptCatalog(contracts)
    cases = json.loads(
        (REPO_ROOT / "dataset/research_cases/pilot_cases.json").read_text(
            encoding="utf-8"
        )
    )["cases"]
    case = next(item for item in cases if item["case_id"] == "image-001")
    media = _case_media(case, DEFAULT_CACHE_DIR)
    results: List[Dict[str, Any]] = []
    evidence: Dict[str, Any] = {
        "schema_version": "research-evidence-v1",
        "packet_type": "initial",
        "request_id": None,
        "facts": [
            {
                "evidence_id": f"f-{index}",
                "fact": item["fact"],
                "supporting_detail": item["career_implication"],
                "source_location": item["location_hint"],
                "confidence": 1.0,
            }
            for index, item in enumerate(case["expected_evidence"], start=1)
        ],
        "missing_fields": [],
        "conflicts": [],
    }

    tasks = [(FROZEN_MODELS[0], "plan"), (FROZEN_MODELS[1], "evidence"), (FROZEN_MODELS[2], "plan")]
    selected_tasks = (
        tasks
        if args.model == "all"
        else [task for task in tasks if task[0]["model_id"] == args.model]
    )
    failures = 0
    for entry, kind in selected_tasks:
        client = LocalModelClient(
            entry["model_id"],
            models_dir=Path(args.models_dir),
            device=args.device,
        )
        telemetry: Optional[Dict[str, Any]] = None
        prompt_hash: Optional[str] = None
        try:
            client.load()
            if entry["role"] == "monolithic":
                system, user, prompt_hash = catalog.planning(case, None, raw_media=True)
                telemetry = client.generate(
                    system,
                    user,
                    media_path=media,
                    constraint=build_constraint_spec(
                        contracts,
                        "plan",
                        allowed_evidence_ids=None,
                        allowed_knowledge_ids=[
                            item["knowledge_id"]
                            for item in case["knowledge_snippets"]
                        ],
                    ),
                )
            elif entry["role"] == "perceiver":
                system, user, prompt_hash = catalog.perception_initial(case)
                telemetry = client.generate(
                    system,
                    user,
                    media_path=media,
                    constraint=build_constraint_spec(
                        contracts, "evidence", packet_type="initial"
                    ),
                )
            else:
                system, user, prompt_hash = catalog.planning(case, evidence, raw_media=False)
                telemetry = client.generate(
                    system,
                    user,
                    constraint=build_constraint_spec(
                        contracts,
                        "plan",
                        allowed_evidence_ids=[
                            item["evidence_id"] for item in evidence["facts"]
                        ],
                        allowed_knowledge_ids=[
                            item["knowledge_id"]
                            for item in case["knowledge_snippets"]
                        ],
                    ),
                )
            payload = _strict_payload(telemetry, kind, contracts)
            _validate_smoke_payload(
                entry, payload, str(telemetry["output_text"])
            )
            if kind == "evidence":
                evidence = payload
            elif entry["role"] == "reasoner":
                _evidence_refs_are_valid(payload, evidence, str(telemetry["output_text"]))
            results.append(
                {
                    **telemetry,
                    "input_case": case["case_id"],
                    "prompt_contract_version": catalog.version,
                    "prompt_hash": prompt_hash,
                    "valid": True,
                    "error": None,
                    **_smoke_outcome_fields(None, telemetry),
                    "raw_invalid_output": None,
                    "output": payload,
                }
            )
        except ModelOutputError as exc:
            failures += 1
            results.append(
                {
                    **(telemetry or {}),
                    "model_id": entry["model_id"],
                    "revision": entry["revision"],
                    "input_case": case["case_id"],
                    "prompt_contract_version": catalog.version,
                    "prompt_hash": prompt_hash,
                    "valid": False,
                    "error": exc.code,
                    **_smoke_outcome_fields(exc.code, telemetry),
                    "raw_invalid_output": exc.raw_output,
                    "output": None,
                }
            )
        except StructuredOutputError as exc:
            failures += 1
            results.append(
                {
                    **(telemetry or {}),
                    "model_id": entry["model_id"],
                    "revision": entry["revision"],
                    "input_case": case["case_id"],
                    "prompt_contract_version": catalog.version,
                    "prompt_hash": prompt_hash,
                    "valid": False,
                    "error": "constraint_failed",
                    **_smoke_outcome_fields("constraint_failed", telemetry),
                    "error_type": type(exc).__name__,
                    "raw_invalid_output": None,
                    "output": None,
                }
            )
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            failures += 1
            results.append(
                {
                    **(telemetry or {}),
                    "model_id": entry["model_id"],
                    "revision": entry["revision"],
                    "input_case": case["case_id"],
                    "prompt_contract_version": catalog.version,
                    "prompt_hash": prompt_hash,
                    "valid": False,
                    "error": "inference_failed",
                    **_smoke_outcome_fields("inference_failed", telemetry),
                    "error_type": type(exc).__name__,
                    "raw_invalid_output": None,
                    "output": None,
                }
            )
        finally:
            client.close()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"smoke-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(results)} real smoke records to {output}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(_main())
