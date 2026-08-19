# Architecture

## Research Position

The project tests whether specialized single-modality perceivers coordinated
with a text reasoner can match or exceed one monolithic multimodal model in a
career-planning task. Career planning is the application and evaluation
domain; the independent variable is the system architecture.

## Implemented Completion MVP

```text
text or TXT/MD/CSV/TSV/PDF/DOCX/XLSX
  -> text/document perception
  -> eight fixed user follow-ups
  -> canonical UserProfile
  -> keyword retrieval over 65 career records
  -> DeepSeek-compatible planner or labeled template fallback
  -> 30/90/180-day plan
  -> three-level feedback
  -> redacted SQLite history and JSONL run record
```

`project/assistant_cli.py` is the canonical CLI. `CareerOrchestrator` owns the
flow; `project/core/` owns schemas, retrieval, settings, privacy, logs, and
persistence. Image, audio, video, vector retrieval, FastAPI, and Web remain
optional. Their imports and model loading must not block text/document use.

The DeepSeek client sends `deepseek-v4-flash` with
`thinking={"type":"disabled"}`. Typed errors distinguish configuration,
authentication, balance, rate-limit, timeout, server, HTTP, and invalid
response failures. Only retryable failures are retried.

## Target Research System

```text
raw modality -> specialized perceiver -> coordinator -> text reasoner
     ^                                                |
     +------ bounded evidence clarification ----------+
  -> shared career retrieval -> final plan and evidence trace
```

The controlled comparison has three groups: `Qwen3-VL-8B-Instruct` as the
monolithic baseline, `Qwen3-VL-2B-Instruct` plus
`Qwen3-4B-Instruct-2507` as the one-shot modular system, and the same modular
pair with bounded evidence clarification. DeepSeek remains an external
engineering reference rather than a primary controlled group.

The current system performs one-shot perception. The monolithic adapter and
reasoner-to-perceiver protocol are specified but not implemented. They must
remain behind an experiment entry point and must not change the completion-MVP
default flow.

## Hardware Profiles

- Core MVP: CPU or GPU machine; no local model is required.
- API/Web: core plus FastAPI, authentication, and SSE.
- Team GPU: optional Qwen/Whisper/BGE development with device-aware VRAM
  thresholds rather than a fixed 6GB assumption.
- Laboratory L20 (48 GB, networked Ubuntu): the only target for downloading
  the three research models and running the primary architecture comparison.
  Record the exact Ubuntu, driver, CUDA, Python, PyTorch, and disk state before
  setup because the Ubuntu version is not yet known.

Weights remain under ignored `models/`; runtime artifacts remain under ignored
`data/`. No research weights are downloaded on the current development
machine. Incoming member work under ignored `corwork/` is review material and
must be adapted into the canonical layout.
