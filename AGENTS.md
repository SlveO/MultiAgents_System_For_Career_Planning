# Repository Guidelines

## Scope and Structure

The original approved career-planning proposal is the sole scope. Use `project/assistant_cli.py` as the only product entry, `project/orchestrator.py` for orchestration, `project/core/` for contracts, retrieval and persistence, and `project/agents/` for perception. Tests live in `project/tests/`; small anonymized inputs and career knowledge belong in `dataset/`. Runtime data and model weights remain in ignored `data/` and `models/`.

Do not restore the cancelled architecture comparisons, pilot matrices, paper research, Web/API stack or independent member branches. The accepted cleanup baseline is `a092c045dd02c835e991796ab3140d034c2e7ca7`. Server snapshot `7995c628595300bd67b122b19df7571ec57e138b` is provenance only; never merge it wholesale. The server is the sole development endpoint; publish effective work to main and start subsequent work on short-lived codex/ branches.

## Runtime and Commands

Deploy the CLI on the Ubuntu GPU server. Only the final reasoning agent uses DeepSeek API. Other model-based components must use existing local weights on an explicitly configured CUDA device. Never load models on CPU, fall back to CPU, or download weights implicitly. Parsing and rule-only tests may run without a GPU.

Install `requirements.txt` for offline development. For the server, install a CUDA-matched torch/torchvision pair first, then `requirements-gpu.txt`. Do not replace a validated server environment merely to match local development.

Run `python -m project.assistant_cli --help`, `python -m compileall -q project scripts/models`, and `python -m unittest discover -s project/tests -v`. Mock network and model calls. Report real GPU/API verification separately.

## Style and Tests

Use four spaces, snake_case symbols, and unittest files and methods named `test_*`. Reuse Pydantic contracts in `project/core/schemas.py`. Keep GPU imports lazy. Add focused regression tests for changed behavior; do not remove failing product tests to report success.

## Collaboration and Handoff

Read current guidance and Git status before edits. Preserve unrelated changes and runtime data. Never commit credentials, raw personal inputs, private paths, weights or caches. Commit, push, merge and remote operations need explicit lead authorization after diff and test review.

Use focused conventional commits and a `codex/` branch. The effective task list is `docs/progress.zh-CN.md`; scope is in `docs/completion-plan.md` and `dataset/completion_protocol.json`. Handoffs must name owner, deliverable, deadline or dependency, acceptance criteria and next action. Record confirmed strategy changes with their corresponding contracts and tests. Server takeover is not complete until the receiving agent acknowledges the final published commit and first task.
