from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, UploadFile, File
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

try:
    from ..core.schemas import CareerPlanResponse, FeedbackRequest, TaskRequest
    from ..core.multimodal_pipeline import MultimodalChatPipeline, PipelineError
    from ..core.settings import get_settings
    from ..core.auth import (
        get_current_user,
        get_current_user_optional,
        login_user,
        register_user,
    )
    from ..orchestrator import CareerOrchestrator
except ImportError:
    from project.core.schemas import CareerPlanResponse, FeedbackRequest, TaskRequest
    from project.core.multimodal_pipeline import MultimodalChatPipeline, PipelineError
    from project.core.settings import get_settings
    from project.core.auth import (
        get_current_user,
        get_current_user_optional,
        login_user,
        register_user,
    )
    from project.orchestrator import CareerOrchestrator


app = FastAPI(title="Career Assistant API", version="0.2.0")

settings = get_settings()
origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

multimodal_pipeline = MultimodalChatPipeline()
_orchestrator: CareerOrchestrator | None = None

UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_orchestrator() -> CareerOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = CareerOrchestrator()
    return _orchestrator


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class MultimodalChatRequest(BaseModel):
    session_id: str = Field(default="default", min_length=1)
    user_input: str = Field(..., min_length=1)
    llm_model: str | None = None


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=4, max_length=128)


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/healthz")
def healthz():
    return JSONResponse(
        content={"status": "ok", "service": "career-assistant"},
        media_type="application/json; charset=utf-8",
    )


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

@app.post("/auth/register")
def auth_register(req: RegisterRequest):
    user = register_user(req.username, req.password)
    return JSONResponse(content=user.model_dump(), media_type="application/json; charset=utf-8")


@app.post("/auth/login")
def auth_login(req: LoginRequest):
    result = login_user(req.username, req.password)
    return JSONResponse(content=result, media_type="application/json; charset=utf-8")


# ---------------------------------------------------------------------------
# Upload (authenticated)
# ---------------------------------------------------------------------------

@app.post("/v1/upload")
def upload_file(file: UploadFile = File(...), user_id: str = Depends(get_current_user)):
    safe_name = f"{uuid.uuid4().hex}_{file.filename or 'upload'}"
    dest = UPLOAD_DIR / safe_name
    content = file.file.read()
    dest.write_bytes(content)
    ext = dest.suffix.lower()
    file_type = "document"
    if ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}:
        file_type = "image"
    elif ext in {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"}:
        file_type = "audio"
    elif ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        file_type = "video"
    return JSONResponse(
        content={
            "file_path": str(dest),
            "file_name": file.filename,
            "file_type": file_type,
            "size": len(content),
        },
        media_type="application/json; charset=utf-8",
    )


# ---------------------------------------------------------------------------
# Career planning (authenticated)
# ---------------------------------------------------------------------------

@app.post("/v1/assist", response_model=CareerPlanResponse)
def assist(req: TaskRequest, user_id: str = Depends(get_current_user)):
    try:
        result = get_orchestrator().run(req)
        return JSONResponse(
            content=jsonable_encoder(result),
            media_type="application/json; charset=utf-8",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"assist_failed: {exc}") from exc


@app.post("/v1/assist/stream")
def assist_stream(req: TaskRequest, user_id: str = Depends(get_current_user)):
    def event_gen():
        try:
            for evt in get_orchestrator().run_stream(req):
                yield {
                    "event": evt.get("event", "stage_progress"),
                    "data": json.dumps(evt.get("data", {}), ensure_ascii=False),
                }
        except Exception as exc:
            yield {
                "event": "error",
                "data": json.dumps({"message": str(exc)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_gen(), media_type="text/event-stream; charset=utf-8")


@app.get("/v1/session/{session_id}")
def get_session(session_id: str, user_id: str = Depends(get_current_user)):
    orchestrator = get_orchestrator()
    profile = orchestrator.memory.get_profile(session_id)
    history = orchestrator.memory.get_session_history(session_id)
    return JSONResponse(
        content={"session_id": session_id, "profile": profile, "history": history},
        media_type="application/json; charset=utf-8",
    )


@app.post("/v1/feedback")
def post_feedback(req: FeedbackRequest, user_id: str = Depends(get_current_user)):
    get_orchestrator().memory.append_feedback(req.session_id, req.feedback, req.rating)
    return JSONResponse(content={"ok": True}, media_type="application/json; charset=utf-8")


# ---------------------------------------------------------------------------
# Multimodal chat (optional auth)
# ---------------------------------------------------------------------------

@app.post("/v1/multimodal/chat/stream")
def multimodal_chat_stream(
    req: MultimodalChatRequest, user_id: str | None = Depends(get_current_user_optional)
):
    def event_gen():
        try:
            for evt in multimodal_pipeline.run_stream(
                req.user_input,
                llm_model=req.llm_model,
                session_id=req.session_id,
            ):
                yield {
                    "event": evt.get("event", "token"),
                    "data": json.dumps(evt.get("data", {}), ensure_ascii=False),
                }
        except PipelineError as exc:
            yield {
                "event": "error",
                "data": json.dumps(
                    {"code": exc.code, "message": exc.message, "session_id": req.session_id},
                    ensure_ascii=False,
                ),
            }
        except Exception as exc:
            yield {
                "event": "error",
                "data": json.dumps(
                    {"code": "INTERNAL_ERROR", "message": str(exc), "session_id": req.session_id},
                    ensure_ascii=False,
                ),
            }

    return EventSourceResponse(event_gen(), media_type="text/event-stream; charset=utf-8")


@app.get("/v1/multimodal/chat/session/{session_id}")
def multimodal_chat_session(session_id: str):
    return JSONResponse(
        content={
            "session_id": session_id,
            "history": multimodal_pipeline.get_session_history(session_id),
        },
        media_type="application/json; charset=utf-8",
    )


@app.delete("/v1/multimodal/chat/session/{session_id}")
def clear_multimodal_chat_session(session_id: str):
    multimodal_pipeline.clear_session(session_id)
    return JSONResponse(content={"ok": True, "session_id": session_id}, media_type="application/json; charset=utf-8")


# ---------------------------------------------------------------------------
# Static web frontend (must be last)
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if _STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
