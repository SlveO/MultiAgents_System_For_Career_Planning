# Verification Guide

## Automated Local Checks

From the repository root in PowerShell:

```powershell
python -m compileall -q project
python -m unittest discover -s project/tests -v
python -m project.assistant_cli --help
python -m project.experiments.run_completion_experiments --output-dir data/experiments/manual-check
```

The default experiment must write eight rows to both `results.json` and
`results.csv`, with `brain_backend` set to `offline-fake`.

## Manual DeepSeek Check

One CLI connectivity case has already been run manually and is recorded in
`progress.md`. For a later repeat, put `DEEPSEEK_API_KEY` in the ignored local
`.env`, then run:

```powershell
python -m project.assistant_cli --session-id live-check --goal "获得数据分析实习" --text "会 Python 和 SQL"
python -m project.experiments.run_completion_experiments --live --output-dir data/experiments/live-check
```

Acceptance:

- output uses `served_by: cloud_brain`, not `local_fallback`;
- model is `deepseek-v4-flash` and the mock payload test confirms thinking is disabled;
- `data/logs/runs.jsonl` contains no key, raw path, email, phone, name, or student ID;
- live results are stored separately from offline fake results.

Stop after one CLI case if the goal is only connectivity. The eight-row live
experiment incurs additional API cost.

## API, GPU, and Frontend

API profile:

```powershell
pip install -r requirements-api.txt
python -m project.api.run_api
```

GPU profile: install the PyTorch build matching the machine CUDA version first,
then install `requirements-gpu.txt`. A personal GPU may be used for interface
smoke checks, but research weights and primary runs are restricted to the L20.

On the L20 Ubuntu host, capture the unknown OS version and hardware state before
installing anything:

```bash
cat /etc/os-release
uname -a
nvidia-smi
python3 --version
df -h
```

Create an isolated Python environment, install a CUDA-compatible PyTorch build,
then install `requirements-gpu.txt`. Verify the environment before downloading
weights:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Only after that preflight, download the frozen experiment models into ignored
`models/` on the L20. Do not run model download scripts on the current Windows
machine. Before the 20-case run, verify all three model identifiers, execute a
six-case pilot, compare two- and three-round caps, and freeze one cap. Preserve
the Git commit, prompts, case manifest, model revisions, generation settings,
environment record, and raw metric files for reproducibility.

The L20 check passes when all three groups complete the same pilot cases,
outputs match the shared schema, collaboration never exceeds the configured
cap, and latency/VRAM/errors are recorded. Model download and L20 inference are
manual future checks, not completed by this documentation change.

Frontend changes require:

```powershell
cd web
npm install
npm run build
```

Docker verification uses `docker compose up --build` with API settings supplied
through the environment. Record dependency-blocked checks separately from real
test failures.
