# Research Decision Log

## Purpose

This is the project lead's durable record of confirmed research choices. The
machine-readable values live in `dataset/research_protocol.json` and the three
research output schemas. Future confirmed changes must be appended here; do not
silently rewrite an earlier decision. Mark an old entry `Superseded` and link
the replacement decision when a choice changes.

## Confirmed Decisions

### RD-001: Research Positioning

- **Date:** 2026-08-20
- **Status:** Confirmed
- **Decision:** Test whether multiple smaller, modality-specialized models can
  match or exceed one monolithic multimodal model in career planning. The
  completion MVP remains the demonstrable product path; training, audio and
  production hardening are not prerequisites for the first architecture study.

### RD-002: Groups and Media Boundary

- **Date:** 2026-08-20
- **Status:** Confirmed
- **Decision:** Compare `Qwen3-VL-8B-Instruct`, one-shot
  `Qwen3-VL-2B-Instruct` plus `Qwen3-4B-Instruct-2507`, and the same modular
  pair with bounded clarification. Only the monolithic planner and 2B perceiver
  receive raw media; the 4B reasoner never receives it. DeepSeek remains the
  completion-MVP planner and is not a primary controlled group.

### RD-003: Reproducible Inference

- **Date:** 2026-08-20
- **Status:** Confirmed
- **Decision:** Use greedy decoding, `do_sample=false`, `num_beams=1`, 2,048
  output tokens, seed 42, BF16 and evaluation mode. Remove executable
  temperature and top-p settings. Freeze exact model revisions before the
  pilot, render PDF pages at 144 DPI, and constrain visual tokens to 256–1,280.
  Weights and formal inference are L20 Ubuntu only.

### RD-004: Prompt and Collaboration Contract

- **Date:** 2026-08-20
- **Status:** Confirmed
- **Decision:** Use one common planning core plus minimal raw-media and
  structured-evidence adapters. Initial perception receives only raw media,
  user goal and target role; it does not receive the full profile or knowledge
  snippets. A clarification returns evidence deltas only and preserves
  conflicts. One collaboration round is one evidence request plus one
  perceiver response; the decision call does not count. Models never receive
  group identity, expected evidence or scoring notes. Prompts require zh-CN
  JSON without Markdown or chain-of-thought and are versioned and hashed.

### RD-005: Output and Failure Contract

- **Date:** 2026-08-20
- **Status:** Confirmed
- **Decision:** Keep the canonical MVP planning fields and add schema version,
  evidence status, missing evidence and traceable evidence references. Keep run
  metadata outside model JSON. Formal runs use direct JSON parsing followed by
  Schema validation, with no fence stripping, repair or format retry. An
  invalid result is retained for diagnosis, counted in validity rate and given
  an effective quality score of 1 in the primary analysis.

### RD-006: Evaluation and Round Selection

- **Date:** 2026-08-20
- **Status:** Confirmed
- **Decision:** Two blinded raters score evidence faithfulness, career
  relevance, actionability, completeness and detail adaptation from 1–5.
  Differences greater than one point require adjudication. Select three rounds
  over two only when the six-case paired pilot has no new failures, improves
  evidence mean by at least 0.50 and five-dimension mean by at least 0.25,
  improves at least three cases, and has no evidence drop greater than one
  point. Keep the pilot separate from the 20-case, three-group, 60-run primary
  comparison.

### RD-007: Change Control

- **Date:** 2026-08-20
- **Status:** Confirmed
- **Decision:** A strategy becomes authoritative only after project-lead
  confirmation. Every confirmed change must update this log, the protocol or
  applicable Schema, related human-readable documentation, and contract tests
  in the same reviewed change. Record rationale and replacement links when a
  decision is superseded.

## Pending Values

The exact Hugging Face model revision hashes and the collaborative
`primary_round_cap` remain intentionally unset. Resolve and freeze revisions on
the L20 before the pilot; select and freeze the round cap only after applying
RD-006. These are pending measurements, not permission to alter the protocol.
