# Repository Guidelines

## Project Structure

The canonical Python implementation is under `project/`. Use `project/assistant_cli.py` for the MVP CLI, `project/orchestrator.py` for the planning flow, `project/core/` for schemas, retrieval, settings, privacy, and persistence, and `project/agents/` for optional perception modules. Tests belong in `project/tests/`; versioned knowledge and anonymized cases belong in `dataset/`; React source is in `web/src/`. Runtime logs, databases, uploads, and experiments go in ignored `data/`; model weights go in ignored `models/`. `corwork/` is an ignored local review area for member submissions, not a second source tree.

## Build, Test, and Development Commands

Install the text/document MVP with `pip install -r requirements.txt`. Use `requirements-api.txt` for FastAPI/Web. For GPU work, first install a CUDA-matched PyTorch build, then run `pip install -r requirements-gpu.txt`; this profile includes API and MVP dependencies.

Run `python -m project.assistant_cli --help`, `python -m unittest discover -s project/tests -v`, and `python -m compileall -q project` before handoff. Run the deterministic four-group harness with `python -m project.experiments.run_completion_experiments --output-dir data/experiments`. Frontend changes require `cd web; npm run build`.

Validate the six-case research plumbing with `python -m project.experiments.run_architecture_experiments --round-cap 2 --output-dir data/experiments/architecture`. Its Fake output is interface evidence only; real model execution follows `docs/member-b-l20-agent-handoff.md` on the L20.

## Style and Testing

Use four-space Python indentation, `snake_case` for Python symbols, `PascalCase` for React components, and `camelCase` for TypeScript variables. Keep canonical Pydantic fields defined in `project/core/schemas.py`; adapt incoming legacy fields at boundaries instead of creating parallel schemas. Tests use `unittest`; name files and methods `test_*`. Mock network/model calls in automated tests. Keep optional GPU imports lazy so the core profile remains runnable without `torch`.

## Data, Security, and Collaboration

Never commit `.env`, keys, weights, raw uploads, generated databases, logs, caches, or private local paths. Redact names, student IDs, phones, emails, and account data from outward-facing artifacts. Download research model weights and run the primary architecture experiment only on the L20 Ubuntu host, not on the current Windows checkout. Member work must be selectively ported into `project/` and covered by regression tests; do not merge legacy `src/` trees wholesale. Handoffs must state owner, deliverable, dependency or deadline, acceptance criteria, and next action.

Use focused conventional commits such as `feat:`, `fix:`, `test:`, and `docs:`. Do not commit, push, merge, publish, or modify remote systems without explicit approval after the final diff and test report are reviewed.

Team tasks start from `origin/integration/week1-results`. Create a personal `work/<role>-<task>` branch and do not push task commits directly to the shared integration branch.

After the project lead confirms a research strategy, update `docs/research-decisions.md`, the applicable machine-readable contract, related documentation, and contract tests in the same reviewed change. Preserve superseded decisions in the log instead of deleting their history.
