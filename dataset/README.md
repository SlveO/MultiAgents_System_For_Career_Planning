# Dataset Layout

`dataset/` contains small, versioned, reviewable project inputs. It is
different from `data/`, which is local runtime state and is excluded from Git.

- `career_knowledge_base.json`: career records used by retrieval. Each record
  should have a stable ID, role, skills, suitable majors, transition path, and
  source information.
- `career_coaching_dataset.jsonl`: fixed anonymized cases for regression and
  experiments. Keep one valid JSON object per line.
- `career.json`: legacy career data retained until its consumers are removed.

Do not put API keys, uploaded documents, names, student IDs, phone numbers,
emails, or private local paths in this directory.
