"""Deterministic structured-output projection and lazy LMFE integration.

This module is safe to import without torch, Transformers, or
lm-format-enforcer. Heavy and optional dependencies are imported only when a
real local-model client initializes the constraint backend.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence


BACKEND_NAME = "lm-format-enforcer"
BACKEND_VERSION = "0.11.3"
INTEGRATION_NAME = "project-transformers-v5-prefix-v1"
PROJECTION_VERSION = "research-decoding-schema-v1"
MAX_CONSECUTIVE_WHITESPACES = 12
MAX_UNBOUNDED_ARRAY_LENGTH = 2048

_EVIDENCE_ID = re.compile(r"^f-[1-9][0-9]*$")
_KNOWLEDGE_ID = re.compile(r"^career-[0-9]{3}$")
_REQUEST_ID = re.compile(r"^r-[1-9][0-9]*$")
_UNSUPPORTED_PROJECTION_KEYS = {
    "$id",
    "$schema",
    "description",
    "examples",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "maximum",
    "minimum",
    "multipleOf",
    "title",
    "uniqueItems",
}


class StructuredOutputError(RuntimeError):
    """A structured-output constraint could not be built or enforced."""


@dataclass(frozen=True)
class StructuredOutputSpec:
    """One deterministic decoding schema plus its canonical provenance."""

    kind: str
    projection_type: str
    decoding_schema: Dict[str, Any]
    canonical_schema_sha256: str
    decoding_schema_sha256: str
    dynamic_enumeration_sources: Dict[str, str]
    dynamic_enumeration_counts: Dict[str, int]

    def telemetry(self) -> Dict[str, Any]:
        return {
            "projection_version": PROJECTION_VERSION,
            "projection_type": self.projection_type,
            "canonical_schema_sha256": self.canonical_schema_sha256,
            "decoding_schema_sha256": self.decoding_schema_sha256,
            "dynamic_enumeration_sources": dict(
                self.dynamic_enumeration_sources
            ),
            "dynamic_enumeration_counts": dict(self.dynamic_enumeration_counts),
        }


def _stable_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_stable_json(value)).hexdigest()


def _project_supported_keywords(value: Any) -> Any:
    """Remove Canonical keywords that LMFE 0.11.3 does not enforce."""
    if isinstance(value, list):
        return [_project_supported_keywords(item) for item in value]
    if not isinstance(value, dict):
        return value
    projected: Dict[str, Any] = {}
    for key, item in value.items():
        if key in _UNSUPPORTED_PROJECTION_KEYS:
            continue
        if key in {"if", "then", "else", "prefixItems"}:
            continue
        projected[key] = _project_supported_keywords(item)
    return projected


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _require_ids(values: Sequence[str], pattern: re.Pattern[str], label: str) -> list[str]:
    normalized = _ordered_unique(values)
    invalid = [value for value in normalized if not pattern.fullmatch(value)]
    if invalid:
        raise StructuredOutputError(f"invalid {label} supplied to decoding schema")
    return normalized


def _project_evidence_schema(
    canonical: Dict[str, Any],
    *,
    packet_type: str,
    request_id: Optional[str],
) -> tuple[Dict[str, Any], str, Dict[str, str], Dict[str, int]]:
    if packet_type not in {"initial", "delta"}:
        raise StructuredOutputError("evidence packet_type must be initial or delta")
    if packet_type == "initial":
        if request_id is not None:
            raise StructuredOutputError("initial evidence request_id must be null")
    elif not isinstance(request_id, str) or not _REQUEST_ID.fullmatch(request_id):
        raise StructuredOutputError("delta evidence requires a canonical request_id")

    projected = copy.deepcopy(canonical)
    projected.pop("allOf", None)
    projected["properties"]["packet_type"] = {"const": packet_type}
    projected["properties"]["request_id"] = (
        {"type": "null"}
        if packet_type == "initial"
        else {"type": "string", "const": request_id}
    )
    return (
        _project_supported_keywords(projected),
        f"evidence-{packet_type}-v1",
        (
            {}
            if packet_type == "initial"
            else {"request_id": "current_decision.request_id"}
        ),
        {} if packet_type == "initial" else {"request_id": 1},
    )


def _project_plan_schema(
    canonical: Dict[str, Any],
    *,
    allowed_evidence_ids: Optional[Sequence[str]],
    allowed_knowledge_ids: Sequence[str],
) -> tuple[Dict[str, Any], str, Dict[str, str], Dict[str, int]]:
    projected = copy.deepcopy(canonical)
    definitions = projected.pop("$defs")
    projected.pop("allOf", None)

    milestone = copy.deepcopy(definitions["milestone_base"])
    milestone["properties"]["period"] = {
        "enum": ["30d", "90d", "180d"]
    }
    projected["properties"]["roadmap_30_90_180"] = {
        "type": "array",
        "minItems": 3,
        "maxItems": 3,
        "items": milestone,
    }

    sources: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    evidence_items = projected["properties"]["evidence_used"]
    evidence_ref = evidence_items["items"]["properties"]["evidence_ref"]
    if allowed_evidence_ids is None:
        sources["evidence_ref"] = "canonical_schema.evidence_ref.pattern"
        projection_type = "plan-raw-media-v1"
    else:
        evidence_ids = _require_ids(
            allowed_evidence_ids, _EVIDENCE_ID, "evidence_id"
        )
        sources["evidence_ref"] = "input_evidence.facts[*].evidence_id"
        counts["evidence_ref"] = len(evidence_ids)
        projection_type = "plan-structured-evidence-v1"
        if evidence_ids:
            evidence_ref.clear()
            evidence_ref.update({"type": "string", "enum": evidence_ids})
        else:
            evidence_items["maxItems"] = 0

    knowledge_ids = _require_ids(
        allowed_knowledge_ids, _KNOWLEDGE_ID, "knowledge_id"
    )
    sources["knowledge_ids_used"] = (
        "input_case.knowledge_snippets[*].knowledge_id"
    )
    counts["knowledge_ids_used"] = len(knowledge_ids)
    knowledge_array = projected["properties"]["knowledge_ids_used"]
    if knowledge_ids:
        knowledge_array["items"] = {
            "type": "string",
            "enum": knowledge_ids,
        }
    else:
        knowledge_array["maxItems"] = 0

    return (
        _project_supported_keywords(projected),
        projection_type,
        sources,
        counts,
    )


def build_constraint_spec(
    contracts: Any,
    kind: str,
    *,
    packet_type: Optional[str] = None,
    request_id: Optional[str] = None,
    allowed_evidence_ids: Optional[Sequence[str]] = None,
    allowed_knowledge_ids: Sequence[str] = (),
) -> StructuredOutputSpec:
    """Project one Canonical Schema into the audited LMFE subset."""
    canonical = contracts.schema(kind)
    canonical_hash = contracts.schema_sha256(kind)
    if kind == "evidence":
        if packet_type is None:
            raise StructuredOutputError("evidence projection requires packet_type")
        projected, projection_type, sources, counts = _project_evidence_schema(
            canonical,
            packet_type=packet_type,
            request_id=request_id,
        )
    elif kind == "plan":
        projected, projection_type, sources, counts = _project_plan_schema(
            canonical,
            allowed_evidence_ids=allowed_evidence_ids,
            allowed_knowledge_ids=allowed_knowledge_ids,
        )
    elif kind == "decision":
        projected = _project_supported_keywords(canonical)
        projection_type = "decision-v1"
        sources = {}
        counts = {}
    else:
        raise StructuredOutputError(f"unsupported structured-output kind: {kind}")
    return StructuredOutputSpec(
        kind=kind,
        projection_type=projection_type,
        decoding_schema=projected,
        canonical_schema_sha256=canonical_hash,
        decoding_schema_sha256=_sha256_json(projected),
        dynamic_enumeration_sources=sources,
        dynamic_enumeration_counts=counts,
    )


def _resident_memory_mb() -> Optional[float]:
    try:
        with open("/proc/self/statm", encoding="ascii") as statm:
            resident_pages = int(statm.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / 2**20
    except (OSError, ValueError, IndexError):
        return None


def _tokenizer_fingerprint(tokenizer: Any) -> str:
    digest = hashlib.sha256()
    digest.update(type(tokenizer).__name__.encode("utf-8"))
    digest.update(str(len(tokenizer)).encode("ascii"))
    digest.update(json.dumps(tokenizer.all_special_ids).encode("ascii"))
    digest.update(str(tokenizer.eos_token_id).encode("ascii"))
    for token, token_id in sorted(
        tokenizer.get_vocab().items(), key=lambda item: (item[1], item[0])
    ):
        encoded = token.encode("utf-8")
        digest.update(int(token_id).to_bytes(4, "big", signed=False))
        digest.update(len(encoded).to_bytes(4, "big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


_TOKENIZER_DATA_CACHE: Dict[str, Any] = {}
_TOKENIZER_TELEMETRY_CACHE: Dict[str, Dict[str, Any]] = {}


def _build_tokenizer_data(tokenizer: Any) -> tuple[Any, Dict[str, Any]]:
    try:
        from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData
    except (ImportError, ModuleNotFoundError) as exc:
        raise StructuredOutputError(
            "lm-format-enforcer backend is unavailable"
        ) from exc

    fingerprint_started = time.perf_counter()
    fingerprint = _tokenizer_fingerprint(tokenizer)
    fingerprint_ms = round(
        (time.perf_counter() - fingerprint_started) * 1000, 3
    )
    if fingerprint in _TOKENIZER_DATA_CACHE:
        telemetry = dict(_TOKENIZER_TELEMETRY_CACHE[fingerprint])
        telemetry.update(
            {
                "tokenizer_data_cache_hit": True,
                "tokenizer_fingerprint_ms": fingerprint_ms,
            }
        )
        return _TOKENIZER_DATA_CACHE[fingerprint], telemetry

    before_rss = _resident_memory_mb()
    started = time.perf_counter()
    token_zero = tokenizer.encode("0")[-1]
    special_ids = set(tokenizer.all_special_ids)
    regular_tokens = []
    for token_id in range(len(tokenizer)):
        if token_id in special_ids:
            continue
        decoded_after_zero = tokenizer.decode([token_zero, token_id])[1:]
        decoded_regular = tokenizer.decode([token_id])
        regular_tokens.append(
            (
                token_id,
                decoded_after_zero,
                len(decoded_after_zero) > len(decoded_regular),
            )
        )

    def decode(tokens: list[int]) -> str:
        return tokenizer.decode(tokens).rstrip("�")

    tokenizer_data = TokenEnforcerTokenizerData(
        regular_tokens,
        decode,
        tokenizer.eos_token_id,
        False,
        len(tokenizer),
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    after_rss = _resident_memory_mb()
    telemetry = {
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_length": len(tokenizer),
        "tokenizer_fingerprint_sha256": fingerprint,
        "tokenizer_fingerprint_ms": fingerprint_ms,
        "tokenizer_data_init_ms": elapsed_ms,
        "tokenizer_data_rss_delta_mb": (
            None
            if before_rss is None or after_rss is None
            else round(max(0.0, after_rss - before_rss), 3)
        ),
        "tokenizer_alphabet_size": len(tokenizer_data.tokenizer_alphabet),
        "tokenizer_data_cache_hit": False,
    }
    _TOKENIZER_DATA_CACHE[fingerprint] = tokenizer_data
    _TOKENIZER_TELEMETRY_CACHE[fingerprint] = dict(telemetry)
    return tokenizer_data, telemetry


class _FailClosedPrefixAllowedTokens:
    def __init__(self, token_enforcer: Any, eos_token_id: Any, force_stop: type) -> None:
        self.token_enforcer = token_enforcer
        self.eos_token_ids = set(
            eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
        )
        self.force_stop = force_stop

    def __call__(self, batch_id: int, sent: Any) -> list[int]:
        del batch_id
        sequence = sent.tolist() if hasattr(sent, "tolist") else list(sent)
        allowed = list(
            self.token_enforcer.get_allowed_tokens(sequence).allowed_tokens
        )
        state = self.token_enforcer.prefix_states.get(tuple(sequence))
        if state is None or isinstance(state.parser, self.force_stop):
            raise StructuredOutputError(
                "constraint parser entered an invalid force-stop state"
            )
        if not allowed:
            raise StructuredOutputError("constraint backend returned no tokens")
        if set(allowed).issubset(self.eos_token_ids) and not state.parser.can_end():
            raise StructuredOutputError(
                "constraint backend terminated before a valid document"
            )
        return allowed


class LMFormatEnforcerBackend:
    """LMFE token filtering with a Transformers 5 compatible prefix adapter."""

    def __init__(self, tokenizer: Any) -> None:
        try:
            installed = importlib.metadata.version(BACKEND_NAME)
        except importlib.metadata.PackageNotFoundError as exc:
            raise StructuredOutputError(
                "lm-format-enforcer backend is not installed"
            ) from exc
        if installed != BACKEND_VERSION:
            raise StructuredOutputError(
                f"lm-format-enforcer version mismatch: expected {BACKEND_VERSION}"
            )
        self.version = installed
        self.tokenizer_data, self.initialization = _build_tokenizer_data(tokenizer)

    def prepare(
        self, spec: StructuredOutputSpec
    ) -> tuple[_FailClosedPrefixAllowedTokens, Dict[str, Any]]:
        try:
            from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser
            from lmformatenforcer.characterlevelparser import ForceStopParser
            from lmformatenforcer.tokenenforcer import TokenEnforcer
        except (ImportError, ModuleNotFoundError) as exc:
            raise StructuredOutputError(
                "lm-format-enforcer backend import failed"
            ) from exc

        before_rss = _resident_memory_mb()
        started = time.perf_counter()
        try:
            config = CharacterLevelParserConfig(
                alphabet=self.tokenizer_data.tokenizer_alphabet,
                max_consecutive_whitespaces=MAX_CONSECUTIVE_WHITESPACES,
                force_json_field_order=False,
                max_json_array_length=MAX_UNBOUNDED_ARRAY_LENGTH,
            )
            parser = JsonSchemaParser(spec.decoding_schema, config=config)
            token_enforcer = TokenEnforcer(self.tokenizer_data, parser)
        except Exception as exc:
            raise StructuredOutputError(
                f"constraint compilation failed for {spec.kind}"
            ) from exc
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        after_rss = _resident_memory_mb()
        prefix = _FailClosedPrefixAllowedTokens(
            token_enforcer,
            self.tokenizer_data.eos_token_id,
            ForceStopParser,
        )
        telemetry = {
            "enabled": True,
            "backend": BACKEND_NAME,
            "backend_version": self.version,
            "integration": INTEGRATION_NAME,
            "constraint_compile_ms": elapsed_ms,
            "constraint_compile_rss_delta_mb": (
                None
                if before_rss is None or after_rss is None
                else round(max(0.0, after_rss - before_rss), 3)
            ),
            "fallback_used": False,
            "repair_used": False,
            **self.initialization,
            **spec.telemetry(),
        }
        return prefix, telemetry
