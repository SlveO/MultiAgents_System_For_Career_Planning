# Experiment Design

## Implemented Completion Checks

`python -m project.experiments.run_completion_experiments` emits eight rows
for four two-variant checks: template vs planner path, no retrieval vs
retrieval, text vs text plus document, and no follow-up vs eight follow-ups.
The default fake client verifies plumbing and result files only; fake or
fallback output is not model-quality evidence.

## Research Questions

- **H1:** Can smaller specialized perceivers plus a text reasoner match or
  exceed a monolithic multimodal model while using less model capacity or
  fewer resources?
- **H2:** Does bounded reasoner-to-perceiver evidence clarification improve
  evidence faithfulness and plan quality over one-shot perception?

## Controlled Architecture Comparison

Use the same fixed user profile, career-knowledge snippets, output schema,
length limit, non-thinking mode, and deterministic generation settings:

1. **Monolithic:** `Qwen3-VL-8B-Instruct` receives the profile, raw image or
   PDF page, and fixed knowledge snippets directly.
2. **Modular one-shot:** `Qwen3-VL-2B-Instruct` extracts evidence once, then
   `Qwen3-4B-Instruct-2507` produces the plan.
3. **Modular collaborative:** the same 2B perceiver and 4B reasoner use the
   bounded evidence protocol in `interfaces.md` before planning.

DeepSeek remains the completion-MVP planner and an external engineering
reference. It is not a primary controlled group because provider and model
scale would introduce an additional confound.

## Cases and Runs

Freeze 20 anonymized cases: 10 images and 10 PDF pages. Decisive evidence must
exist only in the media, with three to five expected evidence facts per case.
User follow-up answers are embedded in the fixed profile; formal runs do not
ask live user questions. Audio, training, and vector-retrieval changes are out
of scope for the first comparison.

First run a six-case pilot. Test collaboration caps of two and three rounds on
the same pilot cases, then freeze one cap for the full comparison; do not mix
caps within the collaborative group. The default recommendation is two rounds
unless the pilot shows a material evidence-quality gain from three. The frozen
comparison requires 20 cases x 3 groups = 60 deterministic inference runs.

A later turn-depth experiment should hold models and cases fixed while testing
caps `0`, `1`, `2`, and `3`. Report quality gain, latency, VRAM, failure rate,
and the number of rounds actually used separately from the primary architecture
result.

## Evaluation

Two blind raters score evidence correctness/faithfulness, career relevance,
actionability, completeness, and detail adaptation separately. Adjudicate any
dimension differing by more than one point. Automated records include JSON
validity, latency, model and environment versions, case ID, modality, errors,
collaboration rounds, fallback status, and peak VRAM.

## L20 Execution Boundary

Research model downloads and real architecture runs occur only on the
networked NVIDIA L20 Ubuntu host, whose available memory is expected to match
the 48 GB profile. Do not download these weights on the current development
machine. Before installation, record the exact Ubuntu version, driver, CUDA,
Python, PyTorch, free disk, and `nvidia-smi` output. Keep weights under ignored
`models/` and results under ignored `data/experiments/`.

Commands and acceptance checks are in `verification.md`.
