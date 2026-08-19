# Project Progress

Last updated: 2026-08-19

Current stage: Week 1 integration is the shared collaboration baseline on
`origin/integration/week1-results`. One user-run DeepSeek CLI case has produced
a valid redacted cloud log. API/Web and full GPU checks remain. `main` is not
part of this publication and requires a later explicit merge decision.

## Overall Plan

| Phase | Goal | Exit criterion | Status |
|---|---|---|---|
| 0. Scope freeze | Define completion MVP, canonical paths, dependencies, and privacy rules | README, interfaces, and acceptance criteria agree | Complete |
| 1. Completion MVP | Run text/document -> follow-up -> profile -> retrieval -> plan -> feedback -> log | Core flow, 50+ roles, four offline comparisons, tests | Complete |
| 2. Team integration | Adapt four contributors' work into one implementation | No parallel `src/`; integrated tests and clean data contracts | Complete, awaiting review |
| 3. Live evidence | Verify one real DeepSeek case, API/Web, and selected GPU smoke paths | Redacted live log and reproducible environment record | In progress: CLI case complete |
| 4. Research comparison | Compare monolithic, one-shot modular, and bounded multi-turn modular systems | Frozen cases, 60 controlled runs, blind scores, report | Protocol frozen; implementation and cases pending |

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
| Project lead | Canonical MVP, privacy, experiments, repository cleanup, final integration | Member branches by 2026-08-21 | Full tests pass; diff contains no secrets/runtime data | Freeze evaluation contracts and review all three member branches |
| Member A | CLI/session/logging skeleton and smoke-flow idea | Frozen experiment protocol | Adapted pipeline events are redacted and covered by end-to-end/log tests | Build the offline architecture runner and bounded controller |
| Member B | DeepSeek baseline, structured failures, retry/schema tests | Authorized L20 access | Typed errors and retries remain covered; no model downloads occur on Windows | Record L20 environment and prepare local-model adapters |
| Member C | 20 career records, follow-up rules, three retrieval cases | Public sources and redistributable case assets | 65 unique roles and three target-role retrieval cases pass | Verify source URLs and deliver the six-case pilot set |

Incoming files remain locally under ignored `corwork/`. Useful behavior was
ported into `project/` and `dataset/`; legacy `src/`, pytest caches, raw logs,
duplicate dependencies, and old planning documents were not imported.

## Two-Day Execution Plan: 2026-08-20 to 2026-08-21

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

## Verification Evidence

- Default `.venv`: 62 discovered; 60 passed; 2 API tests skipped because the
  core profile intentionally omits FastAPI.
- Existing `agents` Conda environment: all 62 tests passed, including API tests.
- `compileall`, CLI help, and `git diff --check`: passed.
- Offline harness: eight rows written to JSON and CSV.
- User-run live evidence: the 2026-08-18 18:45 UTC record uses
  `deepseek-v4-flash`, `cloud_brain`, `completed`, and seven pipeline events.
  The log scan found no raw path/content keys or configured identifier/API-key
  patterns. The input contained no recognized identifiers, so no redaction
  placeholder was expected; synthetic redaction tests still pass.
- Not run by agreement: complete Qwen/Whisper/BGE model load, L20 batch
  experiment, frontend build (`web/node_modules` absent).

## Frozen Research Protocol

- Groups: `Qwen3-VL-8B-Instruct`; one-shot
  `Qwen3-VL-2B-Instruct` -> `Qwen3-4B-Instruct-2507`; and the same modular
  pair with bounded evidence clarification.
- Cases: 10 images plus 10 PDF pages, each with three to five expected facts;
  decisive evidence is not duplicated in the fixed text profile.
- Pilot: six shared cases with two- and three-round caps. Freeze one cap before
  the 20-case, three-group comparison. Study `0/1/2/3` rounds separately later.
- Execution: download research weights and run real comparisons only on the
  networked 48 GB L20 Ubuntu host. The exact Ubuntu version must be captured
  before setup; no research weights are downloaded on this Windows machine.
- Status: protocol documentation is complete. Experiment adapters, case
  manifest, model downloads, inference runs, and human scores do not yet exist.

## Next Actions

1. Every owner branches from `origin/integration/week1-results` and completes
   the two-day deliverable without writing directly to the shared branch.
2. The project lead reviews the three branches after the 2026-08-21 checkpoint
   and integrates only accepted files with passing checks.
3. After the six-case pilot, freeze one collaboration cap before the 60-run
   primary comparison. Training and audio remain outside the first round.
