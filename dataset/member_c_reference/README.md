# Member C Reference Materials

These four JSON files preserve the member's week 1 and week 2 submissions.
They are reference material, not active runtime inputs or validated evaluation
ground truth. Their contents are unchanged from the local submissions.

- `jobs_w1.json`: initial career descriptions.
- `questions_w1.json`: proposed clarification questions and stopping rules.
- `cases_w1.json`: anonymous example profiles and suggested retrieval results.
- `jobs_detail_w2.json`: revised careers with source leads.

Values such as `verification_status`, `verify_date`, and `supported_fields`
record the contributor's claims, not a new verification by this repository.
Some sources have missing URLs. Expected job rankings and career paths require
review before use; they must not be treated as measured results or expert labels.

## Handoff

- Owner: project lead and server agent.
- Deliverable: reference inputs for career knowledge and requirement guidance.
- Dependency: review against current contracts and check source support before
  any integration; no new deadline or experiment is imposed.
- Acceptance: distinguish source-backed facts from suggestions; resolve conflicts
  with current rules and add focused tests for any adopted behavior.
- Next action: consult only when needed for the existing completion plan. Do not
  revive cancelled experiments or replace the active knowledge base wholesale.
