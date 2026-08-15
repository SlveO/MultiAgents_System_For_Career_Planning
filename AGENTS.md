# Repository Guidelines

## Project Structure

The repository combines a Python service with a Vite/React client. Backend entry points are in `project/main.py` (CLI) and `project/api/` (FastAPI). Shared business logic, routing, settings, authentication, RAG, memory, and multimodal orchestration live in `project/core/`; modality-specific agents are under `project/agents/`. Backend tests are in `project/tests/`, with supporting fixtures and datasets in `dataset/` and `project/resources/`. The frontend source is in `web/src/`, while its build output is `web/dist/`.

## Build, Test, and Development Commands

Install Python dependencies with `pip install -r requirements.txt` (use `requirements-docker.txt` for the container-oriented environment). Run the CLI with `python -m project.main --help`. Start the API with `python -m project.api.run_api`, then open `http://localhost:8000`. Run the backend suite with `python -m unittest discover -s project/tests -v`.

For the frontend, run `cd web; npm install` once, `npm run dev` for Vite development, and `npm run build` for the TypeScript/Vite production build. `docker compose up --build` runs the API container; provide `DEEPSEEK_API_KEY` through the environment.

## Coding Style & Naming

Use four-space indentation in Python and the existing TypeScript/TSX formatting. Prefer clear, descriptive `snake_case` names for Python modules, functions, and variables; use `PascalCase` for React components and `camelCase` for frontend functions and variables. Keep API schemas and settings changes localized to their existing modules. No repository-wide formatter or linter is configured, so keep diffs small and manually consistent with nearby code.

## Testing Guidelines

Tests use Python's standard `unittest` framework. Name test files `test_*.py` and test methods `test_*`. Add regression coverage beside the affected subsystem, especially for routing, multimodal flows, API behavior, and persistence. Run the full discovery command before submitting changes; frontend validation currently relies on `npm run build`.

## Commits & Pull Requests

Use imperative, conventional prefixes consistent with history, such as `feat:`, `fix:`, and `docs:`; keep each commit focused. Pull requests should describe behavior changes, list verification commands, identify configuration or migration requirements, and include screenshots for visible frontend changes. Never commit `.env`, API keys, model downloads, generated databases, or temporary test data.

## Configuration & Security

Copy required settings into a local `.env`; at minimum configure `DEEPSEEK_API_KEY`, and use a strong `JWT_SECRET_KEY` outside local development. Review `CORS_ORIGINS` before deployment, and avoid logging uploaded content or credentials.
