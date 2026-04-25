from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

try:
    from ..core.schemas import CareerPlanResponse, FeedbackRequest, TaskRequest
    from ..orchestrator import CareerOrchestrator
except ImportError:
    from project.core.schemas import CareerPlanResponse, FeedbackRequest, TaskRequest
    from project.orchestrator import CareerOrchestrator


app = FastAPI(title="Career Assistant API", version="0.1.0")
orchestrator = CareerOrchestrator()


@app.get("/healthz")
def healthz():
    return JSONResponse(
        content={"status": "ok", "service": "career-assistant"},
        media_type="application/json; charset=utf-8",
    )


@app.post("/v1/assist", response_model=CareerPlanResponse)
def assist(req: TaskRequest):
    try:
        result = orchestrator.run(req)
        return JSONResponse(
            content=jsonable_encoder(result),
            media_type="application/json; charset=utf-8",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"assist_failed: {exc}") from exc


@app.post("/v1/assist/stream")
def assist_stream(req: TaskRequest):
    def event_gen():
        try:
            for evt in orchestrator.run_stream(req):
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
def get_session(session_id: str):
    profile = orchestrator.memory.get_profile(session_id)
    history = orchestrator.memory.get_session_history(session_id)
    return JSONResponse(
        content={"session_id": session_id, "profile": profile, "history": history},
        media_type="application/json; charset=utf-8",
    )


@app.post("/v1/feedback")
def post_feedback(req: FeedbackRequest):
    orchestrator.memory.append_feedback(req.session_id, req.feedback, req.rating)
    return JSONResponse(content={"ok": True}, media_type="application/json; charset=utf-8")
