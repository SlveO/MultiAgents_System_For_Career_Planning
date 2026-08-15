# Architecture

## Research Position

The project studies whether a modular system of specialized perceiver agents
and a text reasoner can match or exceed a monolithic multimodal model in a
career-planning scenario. Career planning is the evaluation domain; the main
research variable is the system architecture.

## Current MVP

```text
text/document input
    -> parser or rule-based perception
    -> structured profile and user follow-up
    -> keyword career retrieval
    -> DeepSeek text planning
    -> three-level feedback and redacted JSONL log
```

The MVP is API-first and can run without GPU dependencies. Optional image,
audio, video, vector retrieval, FastAPI, and Web features must fail readably
and must not block the text/document path.

## Target Research System

```text
raw modality
    -> specialized perceiver (vision / audio / document / text)
    -> coordinator
    -> text reasoner requests evidence or clarification
    -> perceiver answers from the original modality
    -> career knowledge retrieval
    -> final career plan
```

The current orchestrator performs one-shot perception. Reasoner-to-perceiver
multi-turn clarification is a later experiment and must not be described as
implemented until its protocol and tests exist.

## Device Profiles

- MVP: CPU or GPU, core dependencies only, DeepSeek API for planning.
- API/Web: MVP plus FastAPI, authentication, and SSE.
- GPU/L20: API profile plus local Qwen/Whisper/BGE and batch experiments.

Local model weights remain untracked under `models/`.
