# Experiment Design

## Primary Hypothesis

In the same career-planning cases, specialized perceivers plus a text reasoner
can provide better grounded and actionable plans than a single model that
receives all supported modalities directly, while remaining modular and easier
to run or replace.

## Primary Comparison

Use the same cases, career knowledge version, user fields, output schema, and
evaluation rubric for all groups:

1. Single multimodal model receives raw text and image/PDF-page input.
2. One-shot modular system sends one perceiver result to the DeepSeek reasoner.
3. Multi-turn modular system allows the reasoner to ask a perceiver for
   evidence clarification before final planning.

Start with image/PDF cases whose decisive information is not duplicated in the
text prompt. Audio and video are extension cases because comparable monolithic
baselines are harder to control.

## Secondary Ablations

- no knowledge retrieval vs keyword/vector retrieval;
- no user follow-up vs rule-based follow-up;
- untuned perceiver vs collaboration-tuned perceiver, only if data and GPU
  time are available;
- fixed output vs feedback-adjusted output as a usability evaluation.

## Metrics

Blind human scores cover information completeness, career relevance,
actionability, evidence faithfulness, and output adaptation. Automated records
include latency, token usage, errors, model versions, input modality, GPU
model, CUDA/PyTorch versions, and peak memory when available.

Each primary group should use at least 20 fixed anonymized cases, retain raw
redacted logs, report failures, and state limitations. Synthetic fallback
outputs must never be counted as live model results.
