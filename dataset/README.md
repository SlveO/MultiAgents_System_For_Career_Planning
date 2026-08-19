# Dataset Layout

`dataset/` contains small, versioned, reviewable inputs. Local databases,
uploads, logs, and experiment outputs belong in ignored `data/`.

- `career_knowledge_base.json`: 65 unique career records used by keyword or
  optional vector retrieval. The loader assigns `career-NNN` IDs from the
  append-only file order. Records may include skills, suitable majors,
  transition paths, salary hints, and source leads.
- `career_retrieval_cases.json`: three anonymized profiles used to check that
  retrieval returns the intended target role.
- `career_coaching_dataset.jsonl`: anonymized coaching conversations, one JSON
  object per line, retained for later experiments.
- `career.json`: legacy career conversations retained until their consumers
  and research value are reviewed.

Source labels added from team research are provenance leads, not proof that
facts or salary ranges have been independently verified. Verify sources and
record URLs before using them as formal experimental ground truth. Append new
roles instead of reordering existing entries so generated IDs remain stable.

Do not store API keys, raw resumes, names, student IDs, phone numbers, emails,
accounts, or private local paths here.
