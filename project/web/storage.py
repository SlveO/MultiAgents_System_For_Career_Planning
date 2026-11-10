"""Strict SQLite persistence, transactional idempotency and owner authorization."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid
from datetime import datetime, timezone

from project.core.evaluation import make_evaluation


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def iso(value: float | None) -> str | None:
    return datetime.fromtimestamp(value, timezone.utc).isoformat() if value is not None else None


def encode(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def fingerprint(method: str, path: str, body) -> str:
    return hashlib.sha256(encode([method, path, body]).encode()).hexdigest()


class WebStore:
    SCHEMA_VERSION = 1
    RETENTION_SECONDS = 30 * 86400
    # Cleaner runs every minute: start one minute early to meet the 24h cap.
    RAW_SECONDS = 86400 - 60
    MAX_FILE_BYTES = 25 * 1024 * 1024
    MAX_FILES = 3
    ALLOWED_EXTENSIONS = {".txt", ".md", ".csv", ".tsv", ".pdf", ".docx", ".xlsx", ".png", ".jpg", ".jpeg", ".webp", ".mp4", ".avi", ".mov", ".mkv"}

    def __init__(self, db_path: str | Path, upload_dir: str | Path, *, clock=time.time):
        self.db_path, self.upload_dir, self.clock = Path(db_path).resolve(), Path(upload_dir).resolve(), clock

    @contextlib.contextmanager
    def connection(self):
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=10, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=10000")
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except (sqlite3.Error, OSError) as exc:
            if conn:
                conn.rollback()
            raise ApiError(503, "storage_unavailable", "网页存储暂不可用，请稍后重试。") from exc
        except BaseException:
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def initialize(self):
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.upload_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            with self.connection() as conn:
                version = conn.execute("PRAGMA user_version").fetchone()[0]
                if version > self.SCHEMA_VERSION:
                    raise ApiError(503, "schema_incompatible", "网页数据库版本不兼容。")
                for statement in (
                    "CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, owner TEXT NOT NULL, title TEXT NOT NULL, state TEXT NOT NULL, state_version INTEGER NOT NULL DEFAULT 1, created REAL NOT NULL, updated REAL NOT NULL, deleted REAL)",
                    "CREATE INDEX IF NOT EXISTS sessions_owner ON sessions(owner, created)",
                    "CREATE TABLE IF NOT EXISTS uploads (id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), name TEXT NOT NULL, path TEXT NOT NULL, size INTEGER NOT NULL, subject TEXT NOT NULL, created REAL NOT NULL, deleted REAL)",
                    "CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), owner TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL, status TEXT NOT NULL, error TEXT, created REAL NOT NULL, updated REAL NOT NULL)",
                    "CREATE INDEX IF NOT EXISTS tasks_status ON tasks(status, created)",
                    "CREATE TABLE IF NOT EXISTS idempotency (owner TEXT NOT NULL, key TEXT NOT NULL, fingerprint TEXT NOT NULL, response TEXT NOT NULL, status INTEGER NOT NULL, session_id TEXT, operation TEXT NOT NULL, created REAL NOT NULL, PRIMARY KEY(owner,key))",
                    "CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), event TEXT NOT NULL, payload TEXT NOT NULL, created REAL NOT NULL)",
                    "CREATE TABLE IF NOT EXISTS evaluations (id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), plan_id TEXT NOT NULL, output_version INTEGER NOT NULL, payload TEXT NOT NULL, created REAL NOT NULL)",
                ):
                    conn.execute(statement)
                conn.execute(f"PRAGMA user_version={self.SCHEMA_VERSION}")
            os.chmod(self.db_path, 0o600)
        except OSError as exc:
            raise ApiError(503, "storage_unavailable", "无法初始化网页存储。") from exc

    def _session(self, conn, owner: str, session_id: str):
        row = conn.execute("SELECT * FROM sessions WHERE id=? AND owner=? AND deleted IS NULL AND created>?", (session_id, owner, self.clock() - self.RETENTION_SECONDS)).fetchone()
        if row is None:
            raise ApiError(404, "not_found", "记录不存在或已删除。")
        return row

    def _cached(self, conn, owner, key, digest):
        if not key or len(key) > 128 or any(ord(c) < 33 or ord(c) > 126 for c in key):
            raise ApiError(400, "idempotency_key_required", "请提供有效的 Idempotency-Key（1至128个可见ASCII字符）。")
        row = conn.execute("SELECT * FROM idempotency WHERE owner=? AND key=?", (owner, key)).fetchone()
        if row:
            if row["fingerprint"] != digest:
                raise ApiError(409, "idempotency_conflict", "此提交标识已用于不同请求。")
            if row["session_id"] and row["operation"] != "delete":
                self._session(conn, owner, row["session_id"])
            return json.loads(row["response"]), row["status"]
        return None

    def _save_key(self, conn, owner, key, digest, value, status, session_id, operation):
        conn.execute("INSERT INTO idempotency VALUES (?,?,?,?,?,?,?,?)", (owner, key, digest, encode(value), status, session_id, operation, self.clock()))

    def _event(self, conn, session_id, event, payload=None):
        conn.execute("INSERT INTO events VALUES (?,?,?,?,?)", (uuid.uuid4().hex, session_id, event, encode(payload or {}), self.clock()))

    @staticmethod
    def _task_view(row):
        return {"id": row["id"], "session_id": row["session_id"], "kind": row["kind"], "status": row["status"], "error": json.loads(row["error"]) if row["error"] else None, "created_at": iso(row["created"]), "updated_at": iso(row["updated"])}

    @staticmethod
    def _upload_view(row):
        return {"id": row["id"], "name": row["name"], "size": row["size"], "subject": row["subject"], "created_at": iso(row["created"]), "deleted_at": iso(row["deleted"])}

    def _view(self, conn, row):
        state = json.loads(row["state"])
        task = conn.execute("SELECT * FROM tasks WHERE session_id=? AND status IN ('queued','running') ORDER BY created LIMIT 1", (row["id"],)).fetchone()
        last_task = conn.execute("SELECT * FROM tasks WHERE session_id=? ORDER BY created DESC,rowid DESC LIMIT 1", (row["id"],)).fetchone()
        uploads = conn.execute("SELECT * FROM uploads WHERE session_id=? ORDER BY created", (row["id"],)).fetchall()
        evaluations = conn.execute("SELECT payload FROM evaluations WHERE session_id=? ORDER BY created", (row["id"],)).fetchall()
        result = {"id": row["id"], "title": row["title"], "state_version": row["state_version"], "user_goal": state.get("user_goal", ""), "text_input": state.get("text_input", ""), "status": state.get("status", "collecting"), "question": state.get("question"), "plan": state.get("plan"), "versions": state.get("versions", []), "uploads": [self._upload_view(item) for item in uploads], "active_task": self._task_view(task) if task else None, "created_at": iso(row["created"]), "updated_at": iso(row["updated"]), "evaluations": [json.loads(item[0]) for item in evaluations]}
        for key in ("source_claims", "knowledge_snapshot"):
            if key in state:
                result[key] = state[key]
        result["text_subject"] = state.get("text_subject", "unknown")
        result["warning"] = state.get("warning")
        result["feedback"] = state.get("feedback", [])
        result["last_task"] = self._task_view(last_task) if last_task else None
        return result

    def create_session(self, owner, key, digest, payload):
        with self.connection() as conn:
            cached = self._cached(conn, owner, key, digest)
            if cached:
                return cached
            now, session_id = self.clock(), uuid.uuid4().hex
            state = {**payload, "id": session_id, "status": "collecting", "question": None, "versions": [], "plan": None}
            conn.execute("INSERT INTO sessions VALUES (?,?,?,?,1,?,?,NULL)", (session_id, owner, payload.get("title") or payload["user_goal"][:36], encode(state), now, now))
            self._event(conn, session_id, "create")
            value = self._view(conn, self._session(conn, owner, session_id))
            self._save_key(conn, owner, key, digest, value, 201, session_id, "create")
            return value, 201

    def list_sessions(self, owner):
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM sessions WHERE owner=? AND deleted IS NULL AND created>? ORDER BY updated DESC", (owner, self.clock()-self.RETENTION_SECONDS)).fetchall()
            return {"sessions": [self._view(conn, row) for row in rows]}

    def get_session(self, owner, session_id, *, event="view"):
        with self.connection() as conn:
            row = self._session(conn, owner, session_id)
            if event:
                state = json.loads(row["state"])
                versions = state.get("versions", [])
                self._event(conn, session_id, event, {"plan_id": (state.get("plan") or {}).get("plan_id"), "output_version": versions[-1].get("version") if versions else None})
            return self._view(conn, row)

    def add_upload(self, owner, session_id, key, digest, name, content, subject):
        path = None
        try:
            with self.connection() as conn:
                cached = self._cached(conn, owner, key, digest)
                if cached:
                    return cached
                row = self._session(conn, owner, session_id)
                state = json.loads(row["state"])
                if state.get("extracted") or state.get("status", "collecting") != "collecting":
                    raise ApiError(409, "materials_locked", "材料已解析，请新建会话上传新材料。")
                self._no_active(conn, owner)
                if conn.execute("SELECT COUNT(*) FROM uploads WHERE session_id=?", (session_id,)).fetchone()[0] >= self.MAX_FILES:
                    raise ApiError(413, "file_count_limit", "每个会话最多上传3个文件。")
                suffix = Path(name).suffix.lower()
                if suffix not in self.ALLOWED_EXTENSIONS:
                    raise ApiError(415, "unsupported_file", "此格式暂不支持；首版不接受音频。")
                if not content or len(content) > self.MAX_FILE_BYTES:
                    raise ApiError(413, "file_size_limit", "文件必须非空且不超过25 MiB。")
                upload_id, now = uuid.uuid4().hex, self.clock()
                path = self.upload_dir / (upload_id + suffix)
                with path.open("xb") as handle:
                    os.chmod(path, 0o600)
                    handle.write(content)
                conn.execute("INSERT INTO uploads VALUES (?,?,?,?,?,?,?,NULL)", (upload_id, session_id, name, str(path.resolve()), len(content), subject, now))
                conn.execute("UPDATE sessions SET state_version=state_version+1,updated=? WHERE id=?", (now, session_id))
                self._event(conn, session_id, "upload", {"upload_id": upload_id, "subject": subject})
                value = self._upload_view(conn.execute("SELECT * FROM uploads WHERE id=?", (upload_id,)).fetchone())
                self._save_key(conn, owner, key, digest, value, 201, session_id, "upload")
                return value, 201
        except BaseException:
            if path:
                path.unlink(missing_ok=True)
            raise

    def _no_active(self, conn, owner):
        if conn.execute("SELECT 1 FROM tasks WHERE owner=? AND status IN ('queued','running') LIMIT 1", (owner,)).fetchone():
            raise ApiError(409, "user_task_active", "您已有一个模型任务，请等待完成。")

    def enqueue(self, owner, session_id, key, digest, kind, payload):
        with self.connection() as conn:
            cached = self._cached(conn, owner, key, digest)
            if cached:
                return cached
            row = self._session(conn, owner, session_id)
            state = json.loads(row["state"])
            if payload["state_version"] != row["state_version"]:
                raise ApiError(409, "stale_state", "页面状态已更新，请刷新后重新提交。")
            self._no_active(conn, owner)
            if conn.execute("SELECT COUNT(*) FROM tasks WHERE status='queued'").fetchone()[0] >= 10:
                raise ApiError(429, "queue_full", "当前排队已满，请稍后再试。")
            if kind == "perceive":
                if state.get("extracted") or state.get("status") != "collecting":
                    raise ApiError(409, "materials_locked", "材料已解析，无需重复提交。")
                expired = conn.execute("SELECT 1 FROM uploads WHERE session_id=? AND deleted IS NOT NULL", (session_id,)).fetchone()
                if expired:
                    raise ApiError(409, "materials_expired", "原始材料已过保留期，请新建会话上传。")
            elif kind == "answers":
                question = state.get("question")
                if not question or question.get("id") != payload["question_id"]:
                    raise ApiError(409, "stale_question", "问题已更新，请回答当前问题。")
            elif kind == "plans":
                if not state.get("extracted") or state.get("question") or state.get("plan"):
                    raise ApiError(409, "not_ready", "请先完成材料解析与当前追问。")
            elif kind == "feedback":
                self._validate_version(state, payload["plan_id"], payload["source_version"], latest=True)
            now, task_id = self.clock(), uuid.uuid4().hex
            conn.execute("INSERT INTO tasks VALUES (?,?,?,?,?, 'queued', NULL,?,?)", (task_id, session_id, owner, kind, encode(payload), now, now))
            self._event(conn, session_id, "task_queued", {"task_id": task_id, "kind": kind})
            value = self._task_view(conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone())
            self._save_key(conn, owner, key, digest, value, 202, session_id, kind)
            return value, 202

    def get_task(self, owner, task_id):
        with self.connection() as conn:
            task = conn.execute("SELECT * FROM tasks WHERE id=? AND owner=?", (task_id, owner)).fetchone()
            if task is None:
                raise ApiError(404, "not_found", "任务不存在。")
            self._session(conn, owner, task["session_id"])
            return self._task_view(task)

    def interrupt_running(self):
        with self.connection() as conn:
            conn.execute("UPDATE tasks SET status='interrupted',error=?,updated=? WHERE status='running'", (encode({"code": "server_restarted", "message": "服务重启，任务已中断；未自动重试。"}), self.clock()))

    def claim_task(self):
        with self.connection() as conn:
            if conn.execute("SELECT 1 FROM tasks WHERE status='running' LIMIT 1").fetchone():
                return None
            row = conn.execute("SELECT t.* FROM tasks t JOIN sessions s ON s.id=t.session_id WHERE t.status='queued' AND s.deleted IS NULL AND s.created>? ORDER BY t.created,t.rowid LIMIT 1", (self.clock()-self.RETENTION_SECONDS,)).fetchone()
            if not row:
                return None
            state_row = conn.execute("SELECT state FROM sessions WHERE id=?", (row["session_id"],)).fetchone()
            if row["kind"] == "perceive" and not json.loads(state_row[0]).get("extracted"):
                if conn.execute("SELECT 1 FROM uploads WHERE session_id=? AND deleted IS NOT NULL LIMIT 1", (row["session_id"],)).fetchone():
                    conn.execute("UPDATE tasks SET status='failed',error=?,updated=? WHERE id=?", (encode({"code": "materials_expired", "message": "排队期间原材料已过保留期，请新建会话上传。"}), self.clock(), row["id"]))
                    self._event(conn, row["session_id"], "task_failed", {"task_id": row["id"], "code": "materials_expired"})
                    return None
            conn.execute("UPDATE tasks SET status='running',updated=? WHERE id=?", (self.clock(), row["id"]))
            session = self._session(conn, row["owner"], row["session_id"])
            files = [dict(item) for item in conn.execute("SELECT * FROM uploads WHERE session_id=? AND deleted IS NULL", (row["session_id"],)).fetchall()]
            return {**dict(row), "state": json.loads(session["state"]), "payload": json.loads(row["payload"]), "files": files}

    def finish_task(self, task_id, result=None, error=None):
        paths = []
        with self.connection() as conn:
            task = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
            if not task:
                return
            session = conn.execute("SELECT * FROM sessions WHERE id=? AND deleted IS NULL AND created>?", (task["session_id"], self.clock()-self.RETENTION_SECONDS)).fetchone()
            if not session or task["status"] != "running":
                conn.execute("UPDATE tasks SET status='cancelled',payload='{}',error=NULL,updated=? WHERE id=?", (self.clock(), task_id))
                return
            now = self.clock()
            if result is not None:
                if not isinstance(result, dict):
                    raise ValueError("Engine result must be a dictionary")
                conn.execute("UPDATE sessions SET state=?,state_version=state_version+1,updated=? WHERE id=?", (encode(result), now, session["id"]))
                if result.get("extracted"):
                    paths = [item[0] for item in conn.execute("SELECT path FROM uploads WHERE session_id=? AND deleted IS NULL", (session["id"],))]
                    conn.execute("UPDATE uploads SET deleted=? WHERE session_id=? AND deleted IS NULL", (now, session["id"]))
            if error:
                conn.execute("UPDATE tasks SET status='failed',error=?,updated=? WHERE id=?", (encode(error), now, task_id))
                self._event(conn, session["id"], "task_failed", {"task_id": task_id, "code": error["code"]})
            else:
                if result is None:
                    raise ValueError("Engine result must be a dictionary")
                conn.execute("UPDATE tasks SET status='succeeded',payload='{}',updated=? WHERE id=?", (now, task_id))
                self._event(conn, session["id"], "task_succeeded", {"task_id": task_id, "kind": task["kind"]})
                if task["kind"] == "feedback":
                    submitted = json.loads(task["payload"])
                    self._event(conn, session["id"], "feedback", {key: submitted.get(key) for key in ("plan_id", "source_version", "feedback")})
        self._remove_paths(paths)

    @staticmethod
    def _validate_version(state, plan_id, version, *, latest=False):
        versions = state.get("versions", [])
        match = [item for item in versions if item.get("plan_id") == plan_id and item.get("version") == version]
        if not match or (latest and (not versions or versions[-1] != match[0])):
            raise ApiError(409, "stale_version", "规划版本已更新或不存在，请重新选择。")

    def record(self, owner, session_id, key, digest, payload, *, evaluation=False):
        with self.connection() as conn:
            cached = self._cached(conn, owner, key, digest)
            if cached:
                return cached
            row = self._session(conn, owner, session_id)
            state = json.loads(row["state"])
            if evaluation:
                self._validate_version(state, payload["plan_id"], payload["output_version"])
                value = make_evaluation(payload, created_at=iso(self.clock()))
                conn.execute("INSERT INTO evaluations VALUES (?,?,?,?,?,?)", (value["id"], session_id, payload["plan_id"], payload["output_version"], encode(value), self.clock()))
                self._event(conn, session_id, "evaluation", {"evaluation_id": value["id"], "plan_id": payload["plan_id"], "output_version": payload["output_version"]})
                status = 201
            else:
                if payload.get("plan_id") is not None or payload.get("output_version") is not None:
                    self._validate_version(state, payload.get("plan_id"), payload.get("output_version"))
                self._event(conn, session_id, payload["event"], payload)
                value, status = {"ok": True}, 200
            self._save_key(conn, owner, key, digest, value, status, session_id, "evaluation" if evaluation else "event")
            return value, status

    def delete_session(self, owner, session_id, key, digest):
        paths = []
        with self.connection() as conn:
            cached = self._cached(conn, owner, key, digest)
            if cached:
                return cached
            self._session(conn, owner, session_id)
            paths = self._erase(conn, session_id)
            value = {"deleted": True}
            self._save_key(conn, owner, key, digest, value, 200, session_id, "delete")
        self._remove_paths(paths)
        return value, 200

    def _erase(self, conn, session_id):
        now = self.clock()
        paths = [row[0] for row in conn.execute("SELECT path FROM uploads WHERE session_id=?", (session_id,))]
        conn.execute("UPDATE sessions SET state='{}',title='',deleted=?,updated=? WHERE id=?", (now, now, session_id))
        conn.execute("UPDATE tasks SET status='cancelled',payload='{}',error=NULL,updated=? WHERE session_id=?", (now, session_id))
        conn.execute("DELETE FROM uploads WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM events WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM evaluations WHERE session_id=?", (session_id,))
        # Keep the key's fingerprint as a tombstone so retrying an old create
        # cannot silently create a new session or reveal a deleted response.
        conn.execute("UPDATE idempotency SET response='{}',operation='revoked' WHERE session_id=?", (session_id,))
        return paths

    def _remove_paths(self, paths):
        root = self.upload_dir.resolve()
        for value in paths:
            path = Path(value)
            if path.resolve().parent != root:
                raise ApiError(503, "invalid_storage_path", "上传存储路径无效。")
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                raise ApiError(503, "cleanup_failed", "文件清理失败，记录访问已撤销。") from exc

    def cleanup(self):
        paths, erased = [], 0
        with self.connection() as conn:
            now = self.clock()
            for row in conn.execute("SELECT id FROM sessions WHERE deleted IS NULL AND created<=?", (now-self.RETENTION_SECONDS,)).fetchall():
                paths.extend(self._erase(conn, row[0]))
                erased += 1
            rows = conn.execute("SELECT id,path FROM uploads WHERE deleted IS NULL AND created<=?", (now-self.RAW_SECONDS,)).fetchall()
            for row in rows:
                paths.append(row["path"])
                conn.execute("UPDATE uploads SET deleted=? WHERE id=?", (now, row["id"]))
            # Retry an earlier unlink failure even though access was revoked.
            paths.extend(row[0] for row in conn.execute("SELECT path FROM uploads WHERE deleted IS NOT NULL") if Path(row[0]).exists() and row[0] not in paths)
            conn.execute("DELETE FROM idempotency WHERE created<=?", (now-self.RETENTION_SECONDS,))
            # Tombstones contain only opaque ownership/IDs; remove after retention.
            for row in conn.execute("SELECT id FROM sessions WHERE deleted IS NOT NULL AND deleted<=?", (now-self.RETENTION_SECONDS,)).fetchall():
                conn.execute("DELETE FROM tasks WHERE session_id=?", (row[0],))
                conn.execute("DELETE FROM sessions WHERE id=?", (row[0],))
        self._remove_paths(paths)
        # Remove orphaned upload files after 24h (e.g. process exit before commit).
        try:
            for path in self.upload_dir.iterdir():
                if path.is_file() and path.stat().st_mtime <= self.clock()-self.RAW_SECONDS:
                    path.unlink()
        except OSError as exc:
            raise ApiError(503, "cleanup_failed", "上传目录清理失败。") from exc
        return {"sessions_deleted": erased, "files_deleted": len(paths)}
