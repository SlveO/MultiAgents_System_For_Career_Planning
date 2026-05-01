from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

import bcrypt

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from .settings import AppSettings, get_settings

AUTH_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "auth.db"

bearer_scheme = HTTPBearer(auto_error=False)


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


class UserRow(BaseModel):
    id: str
    username: str
    api_key: str
    created_at: str


def _get_conn() -> sqlite3.Connection:
    AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(AUTH_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            api_key TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _make_api_key() -> str:
    return f"ca-{uuid.uuid4().hex}{uuid.uuid4().hex}"


def create_access_token(user_id: str, settings: AppSettings) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {
        "sub": user_id,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str, settings: AppSettings) -> Optional[str]:
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        return payload.get("sub")
    except JWTError:
        return None


def register_user(username: str, password: str) -> UserRow:
    if not username or not password:
        raise HTTPException(status_code=400, detail="username and password required")
    if len(password) < 4:
        raise HTTPException(status_code=400, detail="password too short (min 4 chars)")

    conn = _get_conn()
    try:
        user_id = uuid.uuid4().hex
        api_key = _make_api_key()
        password_hash = _hash_password(password)
        now = datetime.now(timezone.utc).isoformat()

        conn.execute(
            "INSERT INTO users (id, username, password_hash, api_key, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, username, password_hash, api_key, now),
        )
        conn.commit()
        return UserRow(id=user_id, username=username, api_key=api_key, created_at=now)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="username already exists")


def login_user(username: str, password: str) -> dict:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="invalid username or password")
    if not _verify_password(password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="invalid username or password")

    settings = get_settings()
    token = create_access_token(row["id"], settings)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": row["id"],
        "username": row["username"],
    }


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> str:
    if credentials is None:
        raise HTTPException(status_code=401, detail="authorization header required")
    settings = get_settings()
    user_id = decode_access_token(credentials.credentials, settings)
    if user_id is None:
        raise HTTPException(status_code=401, detail="invalid or expired token")
    return user_id


def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Optional[str]:
    """Optional auth — returns user_id or None without raising 401."""
    if credentials is None:
        return None
    settings = get_settings()
    return decode_access_token(credentials.credentials, settings)
