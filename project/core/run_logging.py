from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .privacy import redact_data
from .schemas import CareerPlanResponse, TaskRequest


class JsonlRunLogger:
    def __init__(self, path: str | Path = "./data/logs/runs.jsonl") -> None:
        self.path = Path(path)

    def log_run(
        self,
        request: TaskRequest,
        response: CareerPlanResponse,
        *,
        feedback: str,
        model_name: str,
        error: str = "",
        feedback_adjusted: bool = False,
    ) -> dict:
        pipeline_events = ["input_received", "perception_completed"]
        if request.follow_up_answers and not request.skip_follow_up:
            pipeline_events.append("follow_up_completed")
        pipeline_events.extend(
            [
                "profile_completed",
                "knowledge_retrieved" if request.use_knowledge else "knowledge_skipped",
                "plan_completed",
                "feedback_recorded",
            ]
        )
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": redact_data(request.session_id),
            "status": "completed",
            "pipeline_events": pipeline_events,
            "input": redact_data(
                {
                    "user_goal": request.user_goal,
                    "text_input": request.text_input,
                    "follow_up_answers": request.follow_up_answers,
                    "document_count": len(request.document_paths),
                    "image_count": len(request.image_paths),
                    "audio_count": len(request.audio_paths),
                    "video_count": len(request.video_paths),
                }
            ),
            "profile": redact_data(response.profile.model_dump()),
            "knowledge_result_ids": list(response.knowledge_hit_ids),
            "model": model_name,
            "output": redact_data(
                {
                    "target_roles": response.target_roles,
                    "gap_analysis": response.gap_analysis,
                    "roadmap_30_90_180": [
                        item.model_dump() for item in response.roadmap_30_90_180
                    ],
                    "next_actions": response.next_actions,
                    "risk_flags": response.risk_flags,
                    "user_facing_advice": response.user_facing_advice,
                    "served_by": response.served_by,
                }
            ),
            "feedback": feedback,
            "latency_ms": response.latency_ms,
        }
        if error:
            record["error"] = error
        if feedback_adjusted:
            record["feedback_adjusted"] = True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record
