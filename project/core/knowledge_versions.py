"""Immutable knowledge releases, explicit review, rollback and retrieval snapshots.

This registry is separate from CLI session storage. It performs keyword retrieval
only and never downloads or loads a model. Review approves publication, not facts.
Run ``python -m project.core.knowledge_versions --help`` for administration.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, ConfigDict, Field

from .career_knowledge import CareerKnowledgeBase, KnowledgeItem, _source_metadata


class KnowledgeRegistryError(RuntimeError):
    pass


class KnowledgeConflict(KnowledgeRegistryError):
    pass


class KnowledgeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9-]{2,127}$")
    role: str = Field(min_length=1, max_length=300)
    skills: list[str] = Field(default_factory=list, max_length=200)
    resources: list[str] = Field(default_factory=list, max_length=200)
    transition_paths: list[str] = Field(default_factory=list, max_length=200)
    salary_hint: str = Field(default="", max_length=2000)
    suitable_majors: list[str] = Field(default_factory=list, max_length=200)
    sources: list[str] = Field(default_factory=list, max_length=200)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_entries(entries: list[dict]) -> list[dict]:
    if not isinstance(entries, list) or not entries:
        raise ValueError("Knowledge release must contain a nonempty list of entries")
    result = [KnowledgeRecord.model_validate(item).model_dump() for item in entries]
    ids = [item["id"] for item in result]
    if len(ids) != len(set(ids)):
        raise ValueError("Knowledge IDs must be unique within a release")
    return result


class _SnapshotKnowledge(CareerKnowledgeBase):
    """Use the existing keyword ranker with immutable in-memory entries."""

    def __init__(self, entries: list[dict]):
        self.use_vector = False
        self.items = [KnowledgeItem(
            item_id=entry["id"], **{key: value for key, value in entry.items() if key != "id"},
        ) for entry in entries]


class KnowledgeRegistry:
    def __init__(self, root: str | Path | None = None, seed_path: str | Path | None = None):
        project_root = Path(__file__).resolve().parents[2]
        self.root = Path(root) if root is not None else project_root / "data" / "web" / "knowledge"
        self.seed_path = Path(seed_path) if seed_path is not None else project_root / "dataset" / "career_knowledge_base.json"
        self.db_path = self.root / "knowledge.sqlite3"
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self._initialize()
        except (OSError, sqlite3.Error) as exc:
            raise KnowledgeRegistryError("Knowledge storage is unavailable") from exc

    @contextlib.contextmanager
    def _connect(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield conn
            conn.commit()
        except (sqlite3.Error, OSError) as exc:
            if conn:
                conn.rollback()
            raise KnowledgeRegistryError("Knowledge storage is unavailable") from exc
        except BaseException:
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def _initialize(self) -> None:
        with self._connect(write=True) as conn:
            for statement in (
                "CREATE TABLE IF NOT EXISTS versions (version_id TEXT PRIMARY KEY, content_sha256 TEXT NOT NULL, entries_json TEXT NOT NULL, created_at TEXT NOT NULL, note TEXT NOT NULL, candidate_id TEXT)",
                "CREATE TABLE IF NOT EXISTS registry_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS candidates (id TEXT PRIMARY KEY, base_version TEXT NOT NULL, entries_json TEXT NOT NULL, content_sha256 TEXT NOT NULL, author TEXT NOT NULL, note TEXT NOT NULL, created_at TEXT NOT NULL, status TEXT NOT NULL, reviewer TEXT, review_note TEXT, reviewed_at TEXT, published_version TEXT)",
                "CREATE TABLE IF NOT EXISTS release_events (id INTEGER PRIMARY KEY AUTOINCREMENT, action TEXT NOT NULL, version_id TEXT NOT NULL, actor TEXT NOT NULL, note TEXT NOT NULL, created_at TEXT NOT NULL)",
            ):
                conn.execute(statement)
            if conn.execute("SELECT value FROM registry_meta WHERE key='current'").fetchone():
                return
            entries = _validate_entries(json.loads(self.seed_path.read_text(encoding="utf-8")))
            serialized = _canonical(entries)
            digest = hashlib.sha256(serialized.encode()).hexdigest()
            version_id = f"kb-{digest[:20]}"
            conn.execute("INSERT INTO versions VALUES (?,?,?,?,?,?)", (
                version_id, digest, serialized, _now(), "固定ID的初始发布；所有原始来源保持未核验", None,
            ))
            conn.execute("INSERT INTO registry_meta VALUES ('current',?)", (version_id,))
            conn.execute("INSERT INTO release_events (action,version_id,actor,note,created_at) VALUES (?,?,?,?,?)", (
                "bootstrap", version_id, "bootstrap", "Existing seed; publication does not verify fields", _now(),
            ))

    @staticmethod
    def _version(row: sqlite3.Row) -> dict:
        result = dict(row)
        serialized = result.pop("entries_json")
        if hashlib.sha256(serialized.encode()).hexdigest() != result["content_sha256"]:
            raise KnowledgeRegistryError("Knowledge release integrity check failed")
        result["entries"] = json.loads(serialized)
        return result

    def current(self) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT v.* FROM versions v JOIN registry_meta m ON m.value=v.version_id WHERE m.key='current'"
            ).fetchone()
            if not row:
                raise KnowledgeRegistryError("Current knowledge release is missing")
            return self._version(row)

    def list_versions(self) -> list[dict]:
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(
                "SELECT version_id,content_sha256,created_at,note,candidate_id FROM versions ORDER BY created_at,version_id"
            ).fetchall()]

    def snapshot(self, query: str, top_k: int = 4) -> dict:
        version = self.current()
        knowledge = _SnapshotKnowledge(version["entries"])
        references = knowledge.retrieve(query, top_k=top_k)
        by_id = {entry["id"]: entry for entry in version["entries"]}
        hits = []
        for reference in references:
            entry = by_id[reference["item_id"]]
            hits.append({
                **entry, "item_id": entry["id"],
                **_source_metadata(entry["sources"]),
                "subject": "job", "verification": "unverified",
            })
        return {
            "version_id": version["version_id"],
            "content_sha256": version["content_sha256"],
            "hits": hits, "references": references,
        }

    def create_candidate(self, entries: list[dict], *, author: str, note: str = "") -> dict:
        entries = _validate_entries(entries)
        if not author.strip():
            raise ValueError("Candidate author is required")
        serialized = _canonical(entries)
        digest = hashlib.sha256(serialized.encode()).hexdigest()
        candidate_id = uuid.uuid4().hex
        with self._connect(write=True) as conn:
            base = conn.execute("SELECT value FROM registry_meta WHERE key='current'").fetchone()[0]
            conn.execute("INSERT INTO candidates (id,base_version,entries_json,content_sha256,author,note,created_at,status) VALUES (?,?,?,?,?,?,?,?)", (
                candidate_id, base, serialized, digest, author.strip(), note, _now(), "pending",
            ))
        return self.get_candidate(candidate_id)

    def get_candidate(self, candidate_id: str) -> dict:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM candidates WHERE id=?", (candidate_id,)).fetchone()
            if not row:
                raise ValueError("Unknown knowledge candidate")
            result = dict(row)
            result["entries"] = json.loads(result.pop("entries_json"))
            return result

    def list_candidates(self) -> list[dict]:
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(
                "SELECT id,base_version,content_sha256,author,note,created_at,status,reviewer,review_note,reviewed_at,published_version FROM candidates ORDER BY created_at,id"
            ).fetchall()]

    def diff_candidate(self, candidate_id: str) -> dict:
        """Show changes by fixed ID so review is independent of list ordering."""
        candidate = self.get_candidate(candidate_id)
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM versions WHERE version_id=?", (candidate["base_version"],)).fetchone()
            if not row:
                raise KnowledgeRegistryError("Candidate base release is missing")
            original = self._version(row)
        before = {entry["id"]: entry for entry in original["entries"]}
        after = {entry["id"]: entry for entry in candidate["entries"]}
        return {
            "base_version": candidate["base_version"],
            "added": [after[key] for key in sorted(after.keys() - before.keys())],
            "removed": [before[key] for key in sorted(before.keys() - after.keys())],
            "changed": [{"id": key, "before": before[key], "after": after[key]}
                        for key in sorted(before.keys() & after.keys()) if before[key] != after[key]],
            "notice": "发布审核不代表来源事实已经核验，原始来源仍保持未核验。",
        }

    def review(self, candidate_id: str, *, reviewer: str, approve: bool, note: str) -> dict:
        if not reviewer.strip() or not note.strip():
            raise ValueError("A named reviewer and review note are required")
        with self._connect(write=True) as conn:
            row = conn.execute("SELECT status FROM candidates WHERE id=?", (candidate_id,)).fetchone()
            if not row:
                raise ValueError("Unknown knowledge candidate")
            if row["status"] != "pending":
                raise KnowledgeConflict("Candidate already reviewed; submit a new candidate for changes")
            conn.execute("UPDATE candidates SET status=?,reviewer=?,review_note=?,reviewed_at=? WHERE id=?", (
                "approved" if approve else "rejected", reviewer.strip(), note.strip(), _now(), candidate_id,
            ))
        return self.get_candidate(candidate_id)

    def publish(self, candidate_id: str, *, actor: str, note: str = "") -> dict:
        if not actor.strip():
            raise ValueError("Publication actor is required")
        with self._connect(write=True) as conn:
            row = conn.execute("SELECT * FROM candidates WHERE id=?", (candidate_id,)).fetchone()
            if not row:
                raise ValueError("Unknown knowledge candidate")
            if row["status"] == "published":
                version = conn.execute("SELECT * FROM versions WHERE version_id=?", (row["published_version"],)).fetchone()
                return self._version(version)
            if row["status"] != "approved":
                raise KnowledgeConflict("Knowledge candidate must be approved before publication")
            current = conn.execute("SELECT value FROM registry_meta WHERE key='current'").fetchone()[0]
            if row["base_version"] != current:
                raise KnowledgeConflict("Current knowledge changed; submit and review a candidate on the current release")
            version_id = f"kb-{row['content_sha256'][:20]}"
            created = _now()
            conn.execute("INSERT OR IGNORE INTO versions VALUES (?,?,?,?,?,?)", (
                version_id, row["content_sha256"], row["entries_json"], created, note or row["note"], candidate_id,
            ))
            conn.execute("UPDATE registry_meta SET value=? WHERE key='current'", (version_id,))
            conn.execute("UPDATE candidates SET status='published',published_version=? WHERE id=?", (version_id, candidate_id))
            conn.execute("INSERT INTO release_events (action,version_id,actor,note,created_at) VALUES (?,?,?,?,?)", (
                "publish", version_id, actor.strip(), note, created,
            ))
            return self._version(conn.execute("SELECT * FROM versions WHERE version_id=?", (version_id,)).fetchone())

    def rollback(self, version_id: str, *, actor: str, note: str) -> dict:
        if not actor.strip() or not note.strip():
            raise ValueError("Rollback actor and reason are required")
        with self._connect(write=True) as conn:
            row = conn.execute("SELECT * FROM versions WHERE version_id=?", (version_id,)).fetchone()
            if not row:
                raise ValueError("Unknown published knowledge version")
            conn.execute("UPDATE registry_meta SET value=? WHERE key='current'", (version_id,))
            conn.execute("INSERT INTO release_events (action,version_id,actor,note,created_at) VALUES (?,?,?,?,?)", (
                "rollback", version_id, actor.strip(), note.strip(), _now(),
            ))
            return self._version(row)

    def release_events(self) -> list[dict]:
        with self._connect() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM release_events ORDER BY id").fetchall()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="职业知识候选审核、发布、回滚与快照（不调用模型）")
    parser.add_argument("--root", type=Path, default=None, help="独立知识注册表目录")
    parser.add_argument("--seed", type=Path, default=None, help="首次初始化使用的固定ID知识JSON")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("current")
    sub.add_parser("versions")
    sub.add_parser("candidates")
    sub.add_parser("events")
    candidate = sub.add_parser("candidate", help="提交完整知识列表为候选；不覆盖已发布版本")
    candidate.add_argument("file", type=Path)
    candidate.add_argument("--author", required=True)
    candidate.add_argument("--note", default="")
    inspect = sub.add_parser("inspect")
    inspect.add_argument("id")
    diff = sub.add_parser("diff")
    diff.add_argument("id")
    review = sub.add_parser("review", help="审核仅批准发布；不提升原始来源的核验等级")
    review.add_argument("id")
    review.add_argument("--reviewer", required=True)
    review.add_argument("--decision", choices=["approve", "reject"], required=True)
    review.add_argument("--note", required=True)
    publish = sub.add_parser("publish")
    publish.add_argument("id")
    publish.add_argument("--actor", required=True)
    publish.add_argument("--note", default="")
    rollback = sub.add_parser("rollback")
    rollback.add_argument("version_id")
    rollback.add_argument("--actor", required=True)
    rollback.add_argument("--note", required=True)
    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("query")
    snapshot.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args(argv)
    registry = KnowledgeRegistry(args.root, args.seed)
    if args.command == "current":
        result = registry.current()
    elif args.command == "versions":
        result = registry.list_versions()
    elif args.command == "candidates":
        result = registry.list_candidates()
    elif args.command == "events":
        result = registry.release_events()
    elif args.command == "candidate":
        result = registry.create_candidate(json.loads(args.file.read_text(encoding="utf-8")), author=args.author, note=args.note)
    elif args.command == "inspect":
        result = registry.get_candidate(args.id)
    elif args.command == "diff":
        result = registry.diff_candidate(args.id)
    elif args.command == "review":
        result = registry.review(args.id, reviewer=args.reviewer, approve=args.decision == "approve", note=args.note)
    elif args.command == "publish":
        result = registry.publish(args.id, actor=args.actor, note=args.note)
    elif args.command == "rollback":
        result = registry.rollback(args.version_id, actor=args.actor, note=args.note)
    else:
        result = registry.snapshot(args.query, args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
