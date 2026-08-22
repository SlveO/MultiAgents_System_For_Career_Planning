# Project Progress

Last updated: 2026-08-22

Current stage: the lead has locally integrated and repaired the accepted parts
of members A and C; these changes have not been committed or pushed. Member B
did not execute on the L20, so that branch is reference code only. L20 download,
smoke, and pilot evidence remain required. `main` is still unchanged.

## Overall Plan

| Phase | Goal | Exit criterion | Status |
|---|---|---|---|
| 0. Scope freeze | Define completion MVP, canonical paths, dependencies, and privacy rules | README, interfaces, and acceptance criteria agree | Complete |
| 1. Completion MVP | Run text/document -> follow-up -> profile -> retrieval -> plan -> feedback -> log | Core flow, 50+ roles, four offline comparisons, tests | Complete |
| 2. Team integration | Adapt four contributors' work into one implementation | No parallel `src/`; integrated tests and clean data contracts | A/C rework complete locally; awaiting diff review |
| 3. Live evidence | Verify one real DeepSeek case, API/Web, and selected GPU smoke paths | Redacted live log and reproducible environment record | In progress: CLI case complete |
| 4. Research comparison | Compare monolithic, one-shot modular, and bounded multi-turn modular systems | Frozen cases, 60 controlled runs, blind scores, report | Offline runner and six cases complete; L20 backend and real runs pending |

## Final Deliverables

- installable text/document MVP with optional API/Web/GPU profiles;
- canonical schemas, 65-role knowledge base, anonymized cases, and redacted logs;
- repeatable completion ablations producing CSV/JSON;
- frozen primary research cases, model/environment records, human rubric, and
  comparison report;
- maintained architecture, interface, verification, and progress documents.

## Four-Person Handoff

| Owner | Integrated deliverable | Dependency/deadline | Acceptance | Next action |
|---|---|---|---|---|
| Project lead | MVP, research contracts, A/C rework, final integration | Review the current local diff | Full tests pass; no secrets, weights, or runtime data | Present the diff/test report and wait for publication approval |
| Member A | Offline three-group runner and bounded controller | Frozen protocol and six-case manifest | Strict Schemas, hashes, case-driven output, and 2/3-round tests pass | Selectively ported and repaired by the lead |
| Member B | L20 downloads, real adapters, smoke, and pilot | Networked NVIDIA L20 Ubuntu host | Environment, revisions, valid outputs, latency, VRAM, and 24 real runs | Follow `member-b-l20-agent-handoff.md`; PC results do not satisfy acceptance |
| Member C | Job-source leads and six pilot fixtures | Public leads and self-created anonymous media | 3 PNG + 3 PDF cases with hashes, licenses, profiles, knowledge, and evidence | Normalized by the lead; field-level source review remains |

Incoming files remain locally under ignored `corwork/`. Useful behavior was
ported into `project/` and `dataset/`; legacy `src/`, pytest caches, raw logs,
duplicate dependencies, and old planning documents were not imported.

## Archived Two-Day Execution Plan: 2026-08-20 to 2026-08-21

This table preserves the original assignment for traceability. Current status
is defined by the handoff table and next actions.

All owners run `git fetch origin` and create the listed branch from
`origin/integration/week1-results`. They must not push directly to the shared
integration branch.

| Owner / branch | Day 1: 2026-08-20 | Day 2: 2026-08-21 | Deliverable | Acceptance and next action |
|---|---|---|---|---|
| Project lead / `work/lead-evaluation` | Freeze the case manifest fields, shared prompts, output schema, and five-dimension 1-5 rubric | Review A/B/C branches, rerun checks, and record accept/revise decisions | `docs/evaluation-rubric.md` plus integration review notes | Rubric has scoring anchors and disagreement handling; no incomplete work is merged |
| Member A / `work/member-a-experiment-runner` | Add dependency-free experiment contracts and fake adapters for all three groups | Add the bounded 2/3-round controller, offline runner, and `unittest` coverage | `project/experiments/architecture_protocol.py`, `run_architecture_experiments.py`, and focused tests | Core import works without `torch`; fake run emits three groups; cap and failure fallback tests pass; push branch for lead review |
| Member B / `work/member-b-l20-models` | On L20 only, record Ubuntu, driver, CUDA, Python, PyTorch, disk, network, and VRAM state | Add reproducible download/adapter scripts, download the three frozen models on L20, and run one deterministic smoke response per model | `scripts/models/download_research_models.py`, local-model adapter, and sanitized `docs/l20-setup.md` | Exact model revisions and peak VRAM recorded; no hostname, account, token, private path, or Windows model download; push code/docs branch |
| Member C / `work/member-c-pilot-cases` | Verify source URLs for the 20 contributed career records and select licensed or self-created pilot inputs | Deliver three image and three PDF-page cases with fixed profiles, knowledge IDs, scoring notes, and 3-5 expected facts each | Versioned source metadata and `dataset/research_cases/` manifest/assets/README | Six cases parse, contain no identity data, and include source/license fields; push branch for lead review |

If L20 access or a dependency blocks a task, the owner records the exact command,
error, completed evidence, and next action instead of substituting unverified
results. Training, audio, the 20-case full run, and direct `main` changes are
outside these two days.

### Project Lead Status

The lead selectively ported and repaired member A's runner and member C's data.
The runner now validates all four Schemas and asset hashes, derives fake output
from six real fixtures, and exposes adapter injection for the L20 backend. The
six cases contain three anonymous PNGs and three single-page PDFs. Member C's
20 job records are retained as provenance leads, not field-level ground truth.
Member B's PC-only result remains blocked pending the dedicated L20 handoff.

## Verification Evidence

- Current `agents` environment: all 89 tests passed, including 16 new A/C tests
  and the API integration tests.
- Updated default `.venv`: 89 discovered, 87 passed, and the 2 FastAPI tests
  skipped as expected. The core profile now includes pure-Python `jsonschema`.
- All four Draft 2020-12 research Schemas passed `jsonschema` meta-schema
  validation in the existing `agents` environment.
- `compileall`, CLI help, and `git diff --check`: passed.
- Offline harness: eight rows written to JSON and CSV.
- User-run live evidence: the 2026-08-18 18:45 UTC record uses
  `deepseek-v4-flash`, `cloud_brain`, `completed`, and seven pipeline events.
  The log scan found no raw path/content keys or configured identifier/API-key
  patterns. The input contained no recognized identifiers, so no redaction
  placeholder was expected; synthetic redaction tests still pass.
- Architecture fake run: 18 rows for six cases and three groups. The two-round
  collaborative fake remains insufficient; the three-round fake collects the
  fourth fixture fact. This is plumbing evidence only.
- Not run: any L20 environment capture, Qwen download/inference, 24-run real
  pilot, complete Qwen/Whisper/BGE load, or frontend build.

## Frozen Research Protocol

- Groups: `Qwen3-VL-8B-Instruct`; one-shot
  `Qwen3-VL-2B-Instruct` -> `Qwen3-4B-Instruct-2507`; and the same modular
  pair with bounded evidence clarification.
- Cases: 10 images plus 10 PDF pages, each with three to five expected facts;
  decisive evidence is not duplicated in the fixed text profile.
- Pilot: six shared cases with two- and three-round caps. Freeze one cap before
  the 20-case, three-group comparison. Study `0/1/2/3` rounds separately later.
- Runtime: greedy decoding, 2,048 output tokens, BF16, 144-DPI PDF rendering,
  and 256–1,280 visual tokens. Exact model revisions remain pending L20 freeze.
- Execution: download research weights and run real comparisons only on the
  networked 48 GB L20 Ubuntu host. The exact Ubuntu version must be captured
  before setup; no research weights are downloaded on this Windows machine.
- Status: protocol contracts, the offline adapter boundary, and the six-case
  pilot manifest exist. The real L20 adapters, remaining 14 cases, model
  downloads, inference records, and human scores do not yet exist.

## Next Actions

1. The lead reviews the current A/C diff and test report before authorizing a
   commit or push.
2. The L20 executor starts from the newly published lead baseline and completes
   environment, revision, download, and three-model smoke checkpoint A.
3. After revision approval, run the 24-run six-case pilot, freeze one cap, then
   prepare the remaining cases and 60-run comparison. Training and audio remain
   outside the first comparison.
