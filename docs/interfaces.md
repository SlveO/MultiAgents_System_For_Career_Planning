# Interfaces

## Current Components

| Component | Input | Output | Failure behavior |
|---|---|---|---|
| `DocumentPerceptionAgent` | supported file path | `PerceptionResult` | readable unsupported/missing dependency result |
| `TextPerceptionAgent` | text | extracted facts | rule fallback with missing fields |
| `CareerKnowledgeBase` | profile/query | ranked career IDs and hints | keyword fallback or empty result |
| `CareerOrchestrator` | `TaskRequest` | `CareerPlanResponse` | template fallback and error metadata |
| `JsonlRunLogger` | request/profile/plan/feedback | redacted JSONL record | never writes raw paths or credentials |

## Separate Interaction Layers

User follow-up collects education, major, skills, interests, target role, time
budget, preference, and constraints. It is distinct from the future internal
perceiver dialogue:

```text
reasoner -> perceiver question -> modality evidence -> reasoner
```

The future protocol should cap internal turns, identify the target perceiver,
record the evidence request, and preserve the original plan if clarification
fails. It must be added behind an experiment flag rather than changing the
MVP default path.
