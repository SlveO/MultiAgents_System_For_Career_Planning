"""FastAPI entry point for a private Sites server-to-server proxy."""
from __future__ import annotations

from contextlib import asynccontextmanager
import hashlib
import hmac
import os
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from project.core.output_preferences import OutputPreferences
from project.core.evaluation import EvaluationSubmission
from .queue import ProcessLock, TaskRunner
from .body_limit import BodyLimitMiddleware
from .storage import ApiError, WebStore, encode, fingerprint


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateSession(Payload):
    title: str = Field(default="", max_length=100)
    user_goal: str = Field(min_length=1, max_length=12000)
    text_input: str = Field(default="", max_length=24000)
    text_subject: Literal["student", "job", "reference", "unknown"] = "unknown"
    output_preferences: OutputPreferences | None = None


class StatePayload(Payload):
    state_version: StrictInt = Field(ge=1)


class AnswerPayload(StatePayload):
    question_id: str = Field(min_length=1, max_length=128)
    answer: str = Field(min_length=1, max_length=2000)


class FeedbackPayload(StatePayload):
    plan_id: str = Field(min_length=1, max_length=128)
    source_version: StrictInt = Field(ge=1)
    feedback: Literal["过短", "合适", "过于详细"]


class EventPayload(Payload):
    event: Literal["view", "restore", "graph_view"]
    plan_id: str | None = Field(default=None, max_length=128)
    output_version: StrictInt | None = Field(default=None, ge=1)


def create_app(*, db_path=None, upload_dir=None, service_token=None, engine_factory=None, start_worker=True, clock=None):
    root = Path(os.environ.get("CAREER_WEB_DATA_DIR", Path(__file__).resolve().parents[2] / "data/web"))
    store = WebStore(db_path or os.environ.get("CAREER_WEB_DB", str(root / "web.sqlite3")), upload_dir or os.environ.get("CAREER_WEB_UPLOADS", str(root / "uploads")), **({"clock": clock} if clock else {}))
    token = service_token if service_token is not None else os.environ.get("CAREER_SERVICE_TOKEN", "")
    runner = TaskRunner(store, engine_factory)
    process_lock = ProcessLock(store.db_path.with_suffix(".worker.lock"))

    @asynccontextmanager
    async def lifespan(app):
        if len(token) < 32 or not token.isascii():
            raise RuntimeError("请通过 CAREER_SERVICE_TOKEN 配置至少32字符的服务端密钥。")
        store.initialize()
        process_lock.acquire()
        try:
            if start_worker:
                runner.start()
            yield
        finally:
            runner.stop()
            process_lock.release()

    app = FastAPI(title="Career Planning Private API", version="1", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store, app.state.runner = store, runner

    @app.exception_handler(ApiError)
    async def api_error(request, exc):
        return JSONResponse({"error": {"code": exc.code, "message": exc.message}}, status_code=exc.status, headers={"Cache-Control": "no-store"})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Validation input can contain raw materials; never return it in errors.
        return JSONResponse({"error": {"code": "invalid_request", "message": "请求字段缺失、格式错误或超出限制。"}}, status_code=422)

    @app.middleware("http")
    async def response_headers(request, call_next):
        length = request.headers.get("content-length")
        if length:
            try:
                if int(length) > WebStore.MAX_FILE_BYTES + 1024 * 1024:
                    return JSONResponse({"error": {"code": "file_size_limit", "message": "文件不得超过25 MiB。"}}, status_code=413)
            except ValueError:
                return JSONResponse({"error": {"code": "invalid_request", "message": "无效的请求大小。"}}, status_code=400)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    def identity(authorization: str | None = Header(default=None), x_career_user: str | None = Header(default=None)):
        supplied = (authorization or "").removeprefix("Bearer ")
        if not authorization or not authorization.startswith("Bearer ") or not supplied.isascii() or not hmac.compare_digest(supplied, token):
            raise ApiError(401, "unauthorized", "需要有效的服务端身份。")
        if not x_career_user or len(x_career_user) > 256 or any(ord(c) < 33 or ord(c) > 126 for c in x_career_user):
            raise ApiError(401, "identity_required", "需要已登录的用户身份。")
        # Persist an opaque owner key, not the platform visitor ID itself.
        return hashlib.sha256(x_career_user.encode()).hexdigest()

    def mutation(request: Request, payload):
        return request.headers.get("Idempotency-Key", ""), fingerprint(request.method, request.url.path, payload)

    def write_result(result):
        value, status = result
        return JSONResponse(value, status_code=status)

    @app.get("/healthz")
    def health():
        with store.connection() as conn:
            conn.execute("SELECT 1 FROM sessions LIMIT 1")
        return {"status": "ok", "queue_concurrency": 1}

    @app.get("/v1/sessions")
    def sessions(owner=Depends(identity)):
        return store.list_sessions(owner)

    @app.post("/v1/sessions")
    def create(payload: CreateSession, request: Request, owner=Depends(identity)):
        data = payload.model_dump(mode="json")
        if not data["user_goal"].strip():
            raise ApiError(422, "invalid_request", "请填写职业规划目标。")
        return write_result(store.create_session(owner, *mutation(request, data), data))

    @app.get("/v1/sessions/{session_id}")
    def session(session_id: str, owner=Depends(identity)):
        return store.get_session(owner, session_id)

    @app.post("/v1/sessions/{session_id}/uploads")
    async def upload(session_id: str, request: Request, file: UploadFile = File(...), subject: Literal["student", "job", "reference", "unknown"] = Form(default="unknown"), owner=Depends(identity)):
        # Check ownership before reading any file bytes in the application.
        store.get_session(owner, session_id, event=None)
        name = Path((file.filename or "file").replace("\\", "/")).name
        name = "".join(c for c in name if ord(c) >= 32)[:180]
        try:
            content = await file.read(WebStore.MAX_FILE_BYTES + 1)
        finally:
            await file.close()
        key, digest = mutation(request, {"name": name, "subject": subject, "sha256": hashlib.sha256(content).hexdigest()})
        return write_result(store.add_upload(owner, session_id, key, digest, name, content, subject))

    def enqueue(kind, session_id, payload, request, owner):
        data = payload.model_dump(mode="json")
        response = write_result(store.enqueue(owner, session_id, *mutation(request, data), kind, data))
        runner.wake.set()
        return response

    @app.post("/v1/sessions/{session_id}/perceive")
    def perceive(session_id: str, payload: StatePayload, request: Request, owner=Depends(identity)):
        return enqueue("perceive", session_id, payload, request, owner)

    @app.post("/v1/sessions/{session_id}/answers")
    def answers(session_id: str, payload: AnswerPayload, request: Request, owner=Depends(identity)):
        return enqueue("answers", session_id, payload, request, owner)

    @app.post("/v1/sessions/{session_id}/plans")
    def plans(session_id: str, payload: StatePayload, request: Request, owner=Depends(identity)):
        return enqueue("plans", session_id, payload, request, owner)

    @app.post("/v1/sessions/{session_id}/feedback")
    def feedback(session_id: str, payload: FeedbackPayload, request: Request, owner=Depends(identity)):
        return enqueue("feedback", session_id, payload, request, owner)

    @app.get("/v1/tasks/{task_id}")
    def task(task_id: str, owner=Depends(identity)):
        return store.get_task(owner, task_id)

    @app.post("/v1/sessions/{session_id}/evaluations")
    def evaluations(session_id: str, payload: EvaluationSubmission, request: Request, owner=Depends(identity)):
        data = payload.model_dump(mode="json")
        return write_result(store.record(owner, session_id, *mutation(request, data), data, evaluation=True))

    @app.post("/v1/sessions/{session_id}/events")
    def events(session_id: str, payload: EventPayload, request: Request, owner=Depends(identity)):
        data = payload.model_dump(mode="json")
        return write_result(store.record(owner, session_id, *mutation(request, data), data))

    @app.get("/v1/sessions/{session_id}/versions")
    def versions(session_id: str, owner=Depends(identity)):
        return {"versions": store.get_session(owner, session_id, event=None)["versions"]}

    @app.get("/v1/sessions/{session_id}/export")
    def export(session_id: str, format: Literal["json", "markdown"] = "json", owner=Depends(identity)):
        view = store.get_session(owner, session_id, event="export")
        if format == "json":
            return Response(encode(view), media_type="application/json", headers={"Content-Disposition": f'attachment; filename="career-{session_id}.json"'})
        plan = view.get("plan") or {}
        text = plan.get("report_markdown")
        if not text:
            raise ApiError(409, "plan_unavailable", "尚无可导出的规划报告。")
        return Response(text, media_type="text/markdown; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="career-{session_id}.md"'})

    @app.delete("/v1/sessions/{session_id}")
    def delete(session_id: str, request: Request, owner=Depends(identity)):
        return write_result(store.delete_session(owner, session_id, *mutation(request, None)))

    app.add_middleware(BodyLimitMiddleware)
    return app
