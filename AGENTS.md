# Repository Guidelines

## Project Structure

The repository combines a Python service with a Vite/React client. Backend entry points are `project/assistant_cli.py` (MVP CLI), `project/main.py` (compatibility wrapper), and `project/api/` (optional FastAPI). Shared business logic, routing, settings, retrieval, memory, and orchestration live in `project/core/`; modality-specific agents are under `project/agents/`. Tests are in `project/tests/`, versioned static inputs are in `dataset/`, optional download helpers are in `scripts/models/`, and frontend source is in `web/src/`. Local runtime data and model weights belong in ignored `data/` and `models/` directories.

## Build, Test, and Development Commands

Install the default MVP with `pip install -r requirements.txt`; use `requirements-api.txt` for FastAPI/Web and `requirements-gpu.txt` after installing a CUDA-matched PyTorch build. Run the CLI with `python -m project.assistant_cli --help`. Start the optional API with `python -m project.api.run_api`. Run tests with `python -m unittest discover -s project/tests -v`, and run the offline experiment harness with `python -m project.experiments.run_completion_experiments --output-dir data/experiments`.

For the frontend, run `cd web; npm install` once, `npm run dev` for Vite development, and `npm run build` for the TypeScript/Vite production build. `docker compose up --build` runs the API container; provide `DEEPSEEK_API_KEY` through the environment.

## Coding Style & Naming

Use four-space indentation in Python and the existing TypeScript/TSX formatting. Prefer clear, descriptive `snake_case` names for Python modules, functions, and variables; use `PascalCase` for React components and `camelCase` for frontend functions and variables. Keep API schemas and settings changes localized to their existing modules. No repository-wide formatter or linter is configured, so keep diffs small and manually consistent with nearby code.

## Testing Guidelines

Tests use Python's standard `unittest` framework. Name test files `test_*.py` and test methods `test_*`. Add regression coverage beside the affected subsystem, especially for routing, multimodal flows, dependency profiles, API behavior, and persistence. Run the full discovery command before submitting changes; frontend validation uses `npm run build`.

## Commits & Pull Requests

Use imperative, conventional prefixes consistent with history, such as `feat:`, `fix:`, and `docs:`; keep each commit focused. Pull requests should describe behavior changes, list verification commands, identify configuration or migration requirements, and include screenshots for visible frontend changes. Never commit `.env`, API keys, model downloads, generated databases, raw uploads, or temporary test data.

## Configuration & Security

Copy required settings into a local `.env`; at minimum configure `DEEPSEEK_API_KEY`, and use a strong `JWT_SECRET_KEY` outside local development. Review `CORS_ORIGINS` before deployment, and avoid logging uploaded content or credentials. The current MVP can run without GPU packages, while the GPU profile enables optional local perception and retrieval experiments.
