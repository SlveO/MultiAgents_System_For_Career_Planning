# Completion MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a CPU-only, repeatable career-planning demonstration that accepts text or documents, performs eight fixed follow-up questions, builds a profile, retrieves career knowledge, generates a 30/90/180-day plan, collects three-level feedback, writes redacted logs, and runs four controlled experiments.

**Architecture:** `project.assistant_cli` is the single completion entry point and delegates planning to `CareerOrchestrator`. New focused core modules own intake questions, privacy redaction/run logging, and experiment evaluation. Image, audio, video, vector retrieval, API, and web features remain optional and cannot block the text/document path.

**Tech Stack:** Python 3.10+, Pydantic 2, httpx, pypdf, python-docx, openpyxl, unittest.

**Spec:** User-provided completion MVP brief in the current conversation.

## Global Constraints

- Use `https://api.deepseek.com`, model `deepseek-v4-flash`, and `thinking={"type":"disabled"}`.
- Never hard-code `DEEPSEEK_API_KEY`.
- Text, TXT, PDF, and DOCX planning must work without torch, CUDA, Whisper, Qwen, ChromaDB, or sentence-transformers.
- Preserve existing user changes; do not reset, commit, push, or alter the remote repository.
- Keep runtime `data/` locally but exclude it from Git.

---

### Task 1: CPU-only planning foundation

**Files:**
- Modify: `project/core/settings.py`, `project/core/brain_client.py`
- Modify: `project/orchestrator.py`, `project/agents/image.py`
- Modify: `project/core/input_router.py`, `project/agents/perception/document_agent.py`
- Create: `requirements-mvp.txt`
- Test: `project/tests/test_completion_mvp.py`

- [ ] Add tests proving the model/thinking payload, stable settings instance, supported document suffixes, and orchestrator construction when optional ML imports fail.
- [ ] Run the focused tests and confirm failures are caused by missing behavior.
- [ ] Implement the minimal settings, payload, lazy-agent, CPU fallback, and extension changes.
- [ ] Run focused and existing tests.

### Task 2: Deterministic intake and structured profile

**Files:**
- Create: `project/core/intake.py`
- Modify: `project/core/schemas.py`, `project/orchestrator.py`, `project/assistant_cli.py`
- Test: `project/tests/test_intake.py`, `project/tests/test_completion_flow.py`

- [ ] Test a stable eight-question sequence and answer-to-profile mapping.
- [ ] Implement question definitions, answer collection, and profile merge.
- [ ] Make `project.assistant_cli` the documented interactive entry point with optional JSON answers for repeatable demos.
- [ ] Verify a fake-DeepSeek end-to-end text/document run.

### Task 3: Privacy, feedback, and run records

**Files:**
- Create: `project/core/privacy.py`, `project/core/run_logging.py`
- Modify: `project/core/schemas.py`, `project/core/session_memory.py`, `project/orchestrator.py`, `project/api/api.py`, `project/assistant_cli.py`
- Test: `project/tests/test_privacy_logging.py`

- [ ] Test redaction of names, phones, email, student/identity numbers, and Windows usernames.
- [ ] Test only `过短`, `合适`, and `过于详细` feedback values.
- [ ] Test UTF-8 JSONL records containing redacted input/profile/output, knowledge IDs, model, feedback, and latency.
- [ ] Implement redacted persistence and chronological/clearable session history.

### Task 4: Career knowledge and experiments

**Files:**
- Modify: `project/core/career_knowledge.py`, `dataset/career_knowledge_base.json`
- Create: `project/experiments/__init__.py`, `project/experiments/run_completion_experiments.py`
- Test: `project/tests/test_experiments.py`

- [ ] Test at least 50 uniquely identified knowledge records and keyword retrieval as the default.
- [ ] Expand the dataset to 50-100 representative roles.
- [ ] Implement four deterministic experiment groups and completeness, personalization, actionability, latency, and feedback metrics.
- [ ] Verify the runner emits JSON and CSV results without requiring a live API key.

### Task 5: Documentation and repository hygiene

**Files:**
- Modify: `README.md`, `AGENTS.md`, `.gitignore`
- Create: `.env.example`, `.codex/config.toml`
- Remove: generated caches and tracked `.pyc` files only.

- [ ] Document one-command installation/run, supported input formats, optional features, experiments, and verified commands.
- [ ] Ignore local data, caches, frontend dependencies/build output, local Claude settings, and Codex personal settings.
- [ ] Run all Python tests, CLI help/smoke checks, experiment runner, and frontend build if dependencies remain available.
- [ ] Inspect final Git status and confirm only intentional source/documentation changes remain.
