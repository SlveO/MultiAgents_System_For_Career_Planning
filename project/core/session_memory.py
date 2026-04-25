from __future__ import annotations

import json
import sqlite3
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

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

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
            conn.commit()

    def _init_fallback(self) -> None:
        if not self.fallback_json.exists():
            payload = {"sessions": {}, "interactions": [], "feedback": []}
            self.fallback_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _load_fallback(self) -> Dict[str, Any]:
        self._init_fallback()
        return json.loads(self.fallback_json.read_text(encoding="utf-8"))

    def _save_fallback(self, payload: Dict[str, Any]) -> None:
        self.fallback_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

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

    def get_session_history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        if self.backend == "json":
            data = self._load_fallback()
            rows = [x for x in data.get("interactions", []) if x.get("session_id") == session_id]
            rows = list(reversed(rows))[:limit]
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
        for row in rows:
            history.append(
                {
                    "created_at": row["created_at"],
                    "request": json.loads(row["request_json"]),
                    "response": json.loads(row["response_json"]),
                }
            )
        return history
