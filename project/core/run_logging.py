from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .privacy import redact_data
from .schemas import CareerPlanResponse, FeedbackAdaptationResult, TaskRequest


class JsonlRunLogger:
    def __init__(self, path: str | Path = "./data/logs/runs.jsonl") -> None:
        self.path = Path(path)

    def _feedback_record(self, request_id: str) -> dict | None:
        if self.path.exists():
            with self.path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        saved = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if saved.get("feedback_adaptation", {}).get("request_id") == request_id:
                        return saved
        return None

    def log_feedback_replay(self, adaptation: FeedbackAdaptationResult, versions: list[dict]) -> dict:
        """Repair a missing log after restart without generating or overwriting data."""
        saved = self._feedback_record(adaptation.request_id)
        if saved:
            return saved
        record = redact_data({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": adaptation.session_id, "plan_id": adaptation.plan_id,
            "status": "recovered_feedback", "feedback": adaptation.feedback,
            "feedback_adaptation": adaptation.model_dump(), "output_versions": versions,
            "pipeline_events": ["feedback_recovered"],
        })
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def log_run(
        self,
        request: TaskRequest,
        response: CareerPlanResponse,
        *,
        feedback: str,
        model_name: str,
        adaptation: FeedbackAdaptationResult | None = None,
    ) -> dict:
        if adaptation:
            # A retry after SQLite succeeded must not append a second feedback event.
            saved = self._feedback_record(adaptation.request_id)
            if saved:
                return saved
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
        if adaptation:
            record["plan_id"] = response.plan_id
            record["feedback_adaptation"] = redact_data(adaptation.model_dump())
            record["output_versions"] = [redact_data(response.output_version.model_dump())]
            if adaptation.output_version.version != response.output_version.version:
                record["output_versions"].append(redact_data(adaptation.output_version.model_dump()))
            if adaptation.status == "adapted":
                record["pipeline_events"].append("output_adapted")
            elif adaptation.status == "failed":
                record["pipeline_events"].append("output_adaptation_failed")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record
