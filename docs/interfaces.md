# Interfaces

## Canonical Data Contracts

`TaskRequest` in `project/core/schemas.py` is the only planning request
contract. It contains `session_id`, `user_goal`, optional text/modality paths,
constraints, `follow_up_answers`, `planner_mode`, and `use_knowledge`.

`UserProfile` uses these canonical fields:

```text
education_stage, major, skills[], interests[], target_role,
preference, main_constraints[], constraints
```

`CareerPlanResponse` returns target roles, gap analysis, three milestones
(`30d`, `90d`, `180d`), resources, next actions, risks, knowledge IDs,
backend, retry count, and latency. Feedback is exactly `过短`, `合适`, or
`过于详细`.

Incoming legacy fields must be normalized at the boundary:

| Incoming field | Canonical field |
|---|---|
| `grade` | `education_stage` and `current_stage` |
| `career_goal` / `career_goals` | `target_role` |
| string `interests` | list `interests` |
| `output_preference` | post-plan three-level feedback |

Do not add a parallel `src/` schema or keep Chinese and English variants of
the same field.

## Component Contracts

| Component | Input | Output | Failure behavior |
|---|---|---|---|
| Text/document perceiver | text or supported path | `PerceptionResult` | readable missing/unsupported result |
| Optional media perceiver | media path and goal | `PerceptionResult` | lazy dependency error; core remains usable |
| `CareerKnowledgeBase` | query, `top_k` | ranked IDs and hints | keyword default; vector fallback optional |
| `DeepSeekBrainClient` | prompt and model | text or stream | typed, non-secret `BrainClientError` |
| `CareerOrchestrator` | `TaskRequest` | `CareerPlanResponse` | retryable retry, then labeled template fallback |
| `JsonlRunLogger` | request, response, feedback | one redacted JSONL run | drops raw paths/content and identifiers |

Run records include timestamp, session ID, status, pipeline events, redacted
input/profile/output, knowledge IDs, model, feedback, and latency. The record
is written after feedback, so `feedback_recorded` is a real completed event.

## Future Internal Dialogue

Reasoner-to-perceiver clarification uses an experiment-only protocol and must
not replace the eight user follow-ups in the MVP. A planner decision is either
final output or an evidence request:

```json
{"action": "final"}
```

```json
{
  "action": "request_evidence",
  "target": "vision",
  "question": "Which requirement is stated in the highlighted region?",
  "required_fields": ["requirement", "location"]
}
```

The perceiver response records `facts`, supporting `evidence`, page or region,
`confidence`, and `missing_fields`. Each request targets one perceiver and
counts as one round. Every turn logs the case ID, round index, request,
response, latency, and error without raw private content.

Pilot runs may cap collaboration at two or three rounds. Freeze one cap before
the 20-case architecture comparison; a later `0/1/2/3`-round experiment studies
turn depth separately. At the cap or after a perception failure, the reasoner
must finalize using available evidence and add an insufficient-evidence item to
`risk_flags` rather than retrying without limit.
