from __future__ import annotations

import json
import sqlite3
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class SessionMemory:
    def __init__(self, db_path: str = "./project/data/session_memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.fallback_json = self.db_path.with_suffix(".json")
        self.backend = "sqlite"
        try:
            self._init_db()
        except Exception:
            self.backend = "json"
            self._init_fallback()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    feedback TEXT NOT NULL,
                    rating INTEGER,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS output_versions (
                    plan_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (plan_id, version)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_adaptations (
                    request_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    plan_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _init_fallback(self) -> None:
        if not self.fallback_json.exists():
            payload = {"sessions": {}, "interactions": [], "feedback": []}
            self.fallback_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _load_fallback(self) -> Dict[str, Any]:
        self._init_fallback()
        return json.loads(self.fallback_json.read_text(encoding="utf-8"))

    def _save_fallback(self, payload: Dict[str, Any]) -> None:
        content = json.dumps(payload, ensure_ascii=False)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.fallback_json.parent,
                prefix=f".{self.fallback_json.name}.", delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.fallback_json)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def get_profile(self, session_id: str) -> Dict[str, Any]:
        if self.backend == "json":
            data = self._load_fallback()
            return data.get("sessions", {}).get(session_id, {}).get("profile_json", {})

        with self._connect() as conn:
            row = conn.execute(
                "SELECT profile_json FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return {}
        return json.loads(row["profile_json"])

    def upsert_profile(self, session_id: str, profile: Dict[str, Any]) -> None:
        if self.backend == "json":
            data = self._load_fallback()
            sessions = data.setdefault("sessions", {})
            sessions[session_id] = {"profile_json": profile, "updated_at": self._now()}
            self._save_fallback(data)
            return

        profile_json = json.dumps(profile, ensure_ascii=False)
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions(session_id, profile_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id)
                DO UPDATE SET profile_json = excluded.profile_json, updated_at = excluded.updated_at
                """,
                (session_id, profile_json, now),
            )
            conn.commit()

    def append_interaction(
        self, session_id: str, request_payload: Dict[str, Any], response_payload: Dict[str, Any]
    ) -> None:
        if self.backend == "json":
            data = self._load_fallback()
            data.setdefault("interactions", []).append(
                {
                    "session_id": session_id,
                    "request_json": request_payload,
                    "response_json": response_payload,
                    "created_at": self._now(),
                }
            )
            self._save_fallback(data)
            return

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO interactions(session_id, request_json, response_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    session_id,
                    json.dumps(request_payload, ensure_ascii=False),
                    json.dumps(response_payload, ensure_ascii=False),
                    self._now(),
                ),
            )
            conn.commit()

    def append_feedback(self, session_id: str, feedback: str, rating: Optional[int]) -> None:
        if self.backend == "json":
            data = self._load_fallback()
            data.setdefault("feedback", []).append(
                {
                    "session_id": session_id,
                    "feedback": feedback,
                    "rating": rating,
                    "created_at": self._now(),
                }
            )
            self._save_fallback(data)
            return

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback(session_id, feedback, rating, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, feedback, rating, self._now()),
            )
            conn.commit()

    def save_output_version(self, session_id: str, version: Dict[str, Any]) -> None:
        """Insert an immutable display version; never replace a stored version."""
        if self.backend == "json":
            data = self._load_fallback()
            versions = data.setdefault("output_versions", [])
            for row in versions:
                payload = row["payload"]
                if (payload["plan_id"], payload["version"]) == (version["plan_id"], version["version"]):
                    if row["session_id"] != session_id or payload != version:
                        raise ValueError("Output version already exists with different content")
                    return
            versions.append({"session_id": session_id, "payload": version})
            self._save_fallback(data)
            return
        with self._connect() as conn:
            self._insert_output_version(conn, session_id, version)

    @staticmethod
    def _insert_output_version(conn, session_id: str, version: Dict[str, Any]) -> None:
        existing = conn.execute(
            "SELECT session_id, payload_json FROM output_versions WHERE plan_id = ? AND version = ?",
            (version["plan_id"], version["version"]),
        ).fetchone()
        if existing:
            if existing["session_id"] != session_id or json.loads(existing["payload_json"]) != version:
                raise ValueError("Output version already exists with different content")
            return
        conn.execute(
            "INSERT INTO output_versions VALUES (?, ?, ?, ?, ?)",
            (version["plan_id"], version["version"], session_id, json.dumps(version, ensure_ascii=False), version["created_at"]),
        )

    def get_output_versions(self, session_id: str, plan_id: str) -> List[Dict[str, Any]]:
        if self.backend == "json":
            rows = self._load_fallback().get("output_versions", [])
            return sorted(
                [row["payload"] for row in rows if row["session_id"] == session_id and row["payload"]["plan_id"] == plan_id],
                key=lambda version: version["version"],
            )
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM output_versions WHERE session_id = ? AND plan_id = ? ORDER BY version",
                (session_id, plan_id),
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def get_latest_output_version(self, session_id: str) -> Optional[Dict[str, Any]]:
        if self.backend == "json":
            rows = [row["payload"] for row in self._load_fallback().get("output_versions", []) if row["session_id"] == session_id]
            return max(rows, key=lambda row: (row["created_at"], row["version"])) if rows else None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM output_versions WHERE session_id = ? ORDER BY created_at DESC, version DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def get_feedback_result(self, request_id: str) -> Optional[Dict[str, Any]]:
        if self.backend == "json":
            return self._load_fallback().get("feedback_adaptations", {}).get(request_id)
        with self._connect() as conn:
            row = conn.execute("SELECT payload_json FROM feedback_adaptations WHERE request_id = ?", (request_id,)).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def save_feedback_result(self, result: Dict[str, Any]) -> bool:
        """Atomically save feedback, outcome and a possible new display version."""
        request_id, session_id = result["request_id"], result["session_id"]
        if self.backend == "json":
            data = self._load_fallback()
            results = data.setdefault("feedback_adaptations", {})
            if request_id in results:
                if results[request_id] != result:
                    raise ValueError("Feedback request already exists with different content")
                return False
            if result["status"] == "adapted":
                version = result["output_version"]
                versions = data.setdefault("output_versions", [])
                if any((row["payload"]["plan_id"], row["payload"]["version"]) == (version["plan_id"], version["version"]) for row in versions):
                    raise ValueError("Output version already exists")
                versions.append({"session_id": session_id, "payload": version})
            results[request_id] = result
            data.setdefault("feedback", []).append({
                "session_id": session_id, "feedback": result["feedback"], "rating": None, "created_at": self._now(),
            })
            self._save_fallback(data)
            return True
        with self._connect() as conn:
            existing = conn.execute("SELECT payload_json FROM feedback_adaptations WHERE request_id = ?", (request_id,)).fetchone()
            if existing:
                if json.loads(existing["payload_json"]) != result:
                    raise ValueError("Feedback request already exists with different content")
                return False
            if result["status"] == "adapted":
                self._insert_output_version(conn, session_id, result["output_version"])
            conn.execute(
                "INSERT INTO feedback_adaptations VALUES (?, ?, ?, ?, ?)",
                (request_id, session_id, result["plan_id"], json.dumps(result, ensure_ascii=False), self._now()),
            )
            conn.execute(
                "INSERT INTO feedback (session_id, feedback, rating, created_at) VALUES (?, ?, ?, ?)",
                (session_id, result["feedback"], None, self._now()),
            )
        return True

    def get_session_history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        if self.backend == "json":
            data = self._load_fallback()
            rows = [x for x in data.get("interactions", []) if x.get("session_id") == session_id]
            rows = rows[-limit:]
            return [
                {
                    "created_at": r.get("created_at"),
                    "request": r.get("request_json", {}),
                    "response": r.get("response_json", {}),
                }
                for r in rows
            ]

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT request_json, response_json, created_at
                FROM interactions
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

        history = []
        for row in reversed(rows):
            history.append(
                {
                    "created_at": row["created_at"],
                    "request": json.loads(row["request_json"]),
                    "response": json.loads(row["response_json"]),
                }
            )
        return history

    def clear_session_history(self, session_id: str) -> None:
        if self.backend == "json":
            data = self._load_fallback()
            data["interactions"] = [
                item
                for item in data.get("interactions", [])
                if item.get("session_id") != session_id
            ]
            self._save_fallback(data)
            return

        with self._connect() as conn:
            conn.execute("DELETE FROM interactions WHERE session_id = ?", (session_id,))
            conn.commit()
