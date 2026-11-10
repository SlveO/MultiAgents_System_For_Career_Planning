"""Resumable web steps over the same perception, guidance and planning core.

Only the API's single durable worker invokes this engine. It has no session
database or run logger; the caller commits state atomically with task results.
"""
from __future__ import annotations

import copy
import os
import re
import uuid
from pathlib import Path

from .guidance import advance_guidance, collect_evidence, submit_guidance_answer
from .intake import initialize_fields
from .knowledge_graph import build_career_graph, render_report
from .knowledge_versions import KnowledgeRegistry
from .output_adaptation import adapt_output
from .privacy import redact_data, redact_text
from .schemas import (
    CareerPlanResponse, GuidanceQuestion, OutputVersion, PerceptionResult,
    TaskRequest, UserProfile,
)
from .source_protocol import SourceClaim


class WorkflowError(RuntimeError):
    def __init__(self, code: str, public_message: str, partial_state: dict | None = None):
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message
        self.partial_state = partial_state


class WebPlanningEngine:
    def __init__(self, orchestrator=None, knowledge_registry=None, *, max_rounds: int | None = None):
        # Lazy GPU imports in existing agents are preserved. No weights load here.
        from project.orchestrator import CareerOrchestrator

        data_root = Path(os.environ.get("CAREER_WEB_DATA_DIR", Path(__file__).resolve().parents[2] / "data/web"))
        self.orchestrator = orchestrator or CareerOrchestrator(persist=False)
        self.registry = knowledge_registry or KnowledgeRegistry(
            root=os.environ.get("CAREER_KNOWLEDGE_DIR") or self.orchestrator.settings.career_knowledge_dir or data_root / "knowledge",
        )
        if self.orchestrator.persist:
            raise ValueError("Web orchestration must disable CLI persistence")
        self.orchestrator.knowledge_registry = self.registry
        # A lost response must never cause an implicit second paid request.
        self.orchestrator.settings = self.orchestrator.settings.model_copy(update={"brain_retry_times": 0})
        self.max_rounds = min(4, self.orchestrator.settings.guidance_max_rounds) if max_rounds is None else max_rounds
        if not 0 <= self.max_rounds <= 4:
            raise ValueError("Web guidance supports at most four questions")

    def process(self, kind: str, state: dict, payload: dict, files: list[dict]) -> dict:
        state = copy.deepcopy(state)
        if kind == "perceive":
            return self._perceive(state, files)
        if "request" not in state or not state.get("extracted"):
            raise WorkflowError("not_prepared", "请先解析材料。")
        if kind == "answers":
            return self._answer(state, payload)
        if kind == "plans":
            return self._plan(state)
        if kind == "feedback":
            return self._feedback(state, payload)
        raise WorkflowError("invalid_task", "不支持的任务类型。")

    @staticmethod
    def _strip_paths(value, replacements: dict[str, str]):
        if isinstance(value, dict):
            return {key: WebPlanningEngine._strip_paths(item, replacements) for key, item in value.items()}
        if isinstance(value, list):
            return [WebPlanningEngine._strip_paths(item, replacements) for item in value]
        if isinstance(value, str):
            for path, source_id in replacements.items():
                value = value.replace(path, f"upload:{source_id}")
            return re.sub(r"/(?:home|tmp)/[^\s\"'<>]+", "[LOCAL_PATH]", value)
        return value

    @staticmethod
    def _save_request(state: dict, req: TaskRequest, profile: UserProfile) -> None:
        # Perception results now contain excerpts and opaque source IDs. Neither
        # refresh nor a later answer needs the deleted original files.
        for field in ("document_paths", "image_paths", "audio_paths", "video_paths"):
            setattr(req, field, [])
        req.guidance = profile.guidance.model_copy(deep=True)
        state["request"] = req.model_dump(mode="json")
        state["profile"] = profile.model_dump(mode="json")

    def _perceive(self, state: dict, files: list[dict]) -> dict:
        if state.get("extracted"):
            raise WorkflowError("already_prepared", "材料已解析，请恢复当前会话。")
        categories = {"document_paths": [], "image_paths": [], "video_paths": []}
        ordered = {"document_paths": [], "image_paths": [], "video_paths": []}
        for upload in files:
            suffix = Path(upload["path"]).suffix.lower()
            if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
                category = "image_paths"
            elif suffix in {".mp4", ".avi", ".mov", ".mkv"}:
                category = "video_paths"
            elif suffix in {".txt", ".md", ".csv", ".tsv", ".pdf", ".docx", ".xlsx"}:
                category = "document_paths"
            else:
                raise WorkflowError("unsupported_file", "此文件格式未开放。")
            categories[category].append(upload["path"])
            ordered[category].append(upload)
        req = TaskRequest(
            session_id=state["id"], user_goal=state["user_goal"], text_input=state.get("text_input", ""),
            output_preferences=state.get("output_preferences"),
            metadata={"scan_pdf": "true", "text_subject": state.get("text_subject", "unknown")}, **categories,
        )
        results = self.orchestrator.perceive(req)
        expected = [None] + [upload for group in ordered.values() for upload in group]
        if len(results) != len(expected):
            raise WorkflowError("perception_contract", "材料解析结果数量异常，请保留任务编号。")
        replacements = {item["path"]: item["id"] for item in files}
        annotated = []
        for result, upload in zip(results, expected):
            result = PerceptionResult.model_validate(self._strip_paths(result.model_dump(), replacements))
            result.subject = upload.get("subject", "unknown") if upload else state.get("text_subject", "unknown")
            result.source_id = upload["id"] if upload else "user_input"
            result.verification = "user_confirmed" if not upload and result.subject == "student" else "unverified"
            for evidence in result.evidence:
                evidence.subject, evidence.verification = result.subject, result.verification
                evidence.source_id = result.source_id
            annotated.append(result)
        # Generated facts and summaries are never copied into claimed abilities.
        # Grounded local guidance can read the student's literal excerpts only.
        profile = UserProfile()
        initialize_fields(profile)
        collect_evidence(profile, req, [result for result in annotated if result.subject == "student"])
        state.update(
            perception_results=[item.model_dump(mode="json") for item in annotated],
            rounds=0, question=None, plan=None, versions=[], extracted=True, status="ready",
        )
        self._save_request(state, req, profile)
        self._claims(state, profile, annotated)
        if any(item.confidence <= 0 for item in annotated):
            state["warning"] = "部分材料未成功解析；已保留结果和限制，可在现有信息上规划或新建会话。"
            raise WorkflowError("perception_failed", state["warning"], state)
        return self._next_question(state, req, profile)

    def _next_question(self, state: dict, req: TaskRequest, profile: UserProfile) -> dict:
        step = advance_guidance(
            req, profile, self.orchestrator.guidance_model,
            round_number=state["rounds"] + 1, max_rounds=self.max_rounds,
            output_fn=lambda message: state.update(warning=message),
        )
        self._save_request(state, req, step.profile)
        self._claims(state, step.profile)
        if step.profile.guidance.stop_reason == "model_error":
            state.update(status="ready", question=None, warning="本地引导失败，已保存材料解析和回答。可依据当前信息生成规划。")
            raise WorkflowError("guidance_failed", state["warning"], state)
        state["question"] = ({"id": step.question_id, **step.question.model_dump(mode="json")} if step.question else None)
        state["status"] = "awaiting_answer" if step.question else "ready"
        return state

    def _answer(self, state: dict, payload: dict) -> dict:
        saved = state.get("question")
        if not saved or saved["id"] != payload.get("question_id"):
            raise WorkflowError("stale_question", "此问题已更新，请刷新会话。")
        req = TaskRequest.model_validate(state["request"])
        question = GuidanceQuestion.model_validate({key: value for key, value in saved.items() if key != "id"})
        profile = submit_guidance_answer(
            req, UserProfile.model_validate(state["profile"]), question, saved["id"], payload["answer"],
            round_number=state["rounds"] + 1,
        )
        state["rounds"] += 1
        state["question"] = None
        if profile.guidance.stop_reason == "user_stopped":
            self._save_request(state, req, profile)
            self._claims(state, profile)
            state["status"] = "ready"
            return state
        return self._next_question(state, req, profile)

    @staticmethod
    def _claims(state: dict, profile: UserProfile, results: list[PerceptionResult] | None = None) -> None:
        claims = []
        evidence = {item.evidence_id: item for item in profile.guidance.evidence}
        for field, field_state in profile.guidance.fields.items():
            for value in field_state.values:
                supports = [evidence[ref] for ref in field_state.evidence_ids if ref in evidence]
                confirmed = bool(supports) and all(item.verification != "unverified" for item in supports)
                claims.append({**SourceClaim(
                    field=field, value=value, subject="student",
                    verification="user_confirmed" if confirmed else "unverified",
                    source_id=supports[0].source_id if supports else "",
                    origin="profile", evidence_ids=field_state.evidence_ids,
                ).model_dump(mode="json"), "status": field_state.status})
        if results is None:
            results = [PerceptionResult.model_validate(item) for item in state.get("perception_results", [])]
        # Preserve non-student references and unconfirmed vision separately.
        for result in results:
            if result.subject != "student" or result.requires_confirmation:
                claims.append(SourceClaim(
                    field="material", value=(result.summary or "未提取到可用信息")[:5000],
                    subject=result.subject, verification=result.verification,
                    source_id=result.source_id, origin=result.modality,
                ).model_dump(mode="json"))
        state["source_claims"] = claims

    def _plan(self, state: dict) -> dict:
        if state.get("question") or state.get("plan"):
            raise WorkflowError("invalid_state", "请先完成引导；已有规划可通过反馈调整。")
        from project.orchestrator import PreparedCareerInput

        req = TaskRequest.model_validate(state["request"])
        profile = UserProfile.model_validate(state["profile"])
        results = [PerceptionResult.model_validate(item) for item in state["perception_results"]]
        prepared = PreparedCareerInput(req.model_dump_json(), results, profile)
        response = self.orchestrator.run(req, prepared=prepared)
        if response.served_by != "cloud_brain":
            raise WorkflowError("planning_failed", "规划请求未成功，未自动重试。请查看任务后再决定是否重新提交。")
        # This privacy-safe canonical object is the object used for all future
        # feedback hashes, exports and graphs. No path or raw file is needed.
        response = self.orchestrator._restore_response(redact_data(response.model_dump(mode="json")))
        snapshot = response.knowledge_snapshot
        graph = build_career_graph(response, snapshot["hits"] if snapshot else [])
        plan = response.model_dump(mode="json")
        plan["graph"] = graph
        plan["report_markdown"] = render_report(response, graph, response.output_version.display_text)
        state.update(
            plan=plan, versions=[response.output_version.model_dump(mode="json")],
            knowledge_snapshot=snapshot, status="planned", question=None,
        )
        return state

    def _feedback(self, state: dict, payload: dict) -> dict:
        if not state.get("plan") or not state.get("versions"):
            raise WorkflowError("no_plan", "还没有可以调整的规划。")
        response = CareerPlanResponse.model_validate(state["plan"])
        baseline = OutputVersion.model_validate(state["versions"][-1])
        if payload.get("plan_id") != response.plan_id or payload.get("source_version") != baseline.version:
            raise WorkflowError("stale_version", "规划版本已更新，请恢复后再反馈。")
        result = adapt_output(
            response, payload["feedback"], payload.get("request_id") or uuid.uuid4().hex,
            self.orchestrator.feedback_model, baseline=baseline,
        )
        if result.status == "failed":
            raise WorkflowError("feedback_failed", "表达调整未通过校验，原版本已保留；未自动重试。")
        if result.status == "adapted":
            state["versions"].append(result.output_version.model_dump(mode="json"))
        state.setdefault("feedback", []).append(result.model_dump(mode="json"))
        # Rebuild only the text portion, using the stored graph and snapshot.
        state["plan"]["report_markdown"] = render_report(response, state["plan"]["graph"], result.output_version.display_text)
        state["status"] = "planned"
        return state
