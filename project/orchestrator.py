from __future__ import annotations

import json
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

try:
    from .core.schemas import CareerPlanResponse, Milestone, PerceptionResult, TaskRequest, UserProfile
    from .core.brain_client import (
        BrainClient,
        BrainClientError,
        BrainResponseError,
        DeepSeekBrainClient,
    )
    from .core.career_knowledge import CareerKnowledgeBase
    from .agents.perception import AudioPerceptionAgent, DocumentPerceptionAgent, ImagePerceptionAgent, TextPerceptionAgent, VideoPerceptionAgent
    from .core.session_memory import SessionMemory
    from .core.settings import get_settings
    from .core.intake import apply_answers_to_profile
    from .core.feedback import FEEDBACK_OPTIONS
    from .core.privacy import redact_data
    from .core.run_logging import JsonlRunLogger
    from .core.planning_prompt import build_planning_prompt
    from .core.feedback_prompt import build_feedback_prompt
except ImportError:
    from project.core.schemas import CareerPlanResponse, Milestone, PerceptionResult, TaskRequest, UserProfile
    from project.core.brain_client import (
        BrainClient,
        BrainClientError,
        BrainResponseError,
        DeepSeekBrainClient,
    )
    from project.core.career_knowledge import CareerKnowledgeBase
    from project.agents.perception import AudioPerceptionAgent, DocumentPerceptionAgent, ImagePerceptionAgent, TextPerceptionAgent, VideoPerceptionAgent
    from project.core.session_memory import SessionMemory
    from project.core.settings import get_settings
    from project.core.intake import apply_answers_to_profile
    from project.core.feedback import FEEDBACK_OPTIONS
    from project.core.privacy import redact_data
    from project.core.run_logging import JsonlRunLogger
    from project.core.planning_prompt import build_planning_prompt
    from project.core.feedback_prompt import build_feedback_prompt
from project.utils.fusion import MultiModalFusion


class CareerOrchestrator:
    def __init__(
        self,
        image_model_path: str = './models/Qwen3-VL-2B-Instruct',
        db_path: str = './data/session_memory.db',
        brain_client: Optional[BrainClient] = None,
        run_logger: Optional[JsonlRunLogger] = None,
    ):
        self.settings = get_settings()
        self.memory = SessionMemory(db_path=db_path)
        self.knowledge = CareerKnowledgeBase()
        self.text_agent = TextPerceptionAgent()
        self.document_agent = DocumentPerceptionAgent()
        self.image_agent = None
        self.audio_agent = None
        self.video_agent = None

        self.cloud_brain = brain_client or DeepSeekBrainClient()
        self.run_logger = run_logger or JsonlRunLogger()
        # Pending tuple keeps the perception/knowledge context of the run so
        # feedback adjustment can rebuild the prompt without re-running the
        # pipeline.
        self._pending_runs: Dict[
            str, Tuple[TaskRequest, CareerPlanResponse, str, List[PerceptionResult], List[str], List[str]]
        ] = {}

        self.image_model_path = image_model_path

    def _persist_response(
        self,
        req: TaskRequest,
        response: CareerPlanResponse,
        model_name: str,
        perception_results: List[PerceptionResult],
        knowledge_hints: List[str],
        knowledge_hit_ids: List[str],
    ) -> None:
        self.memory.upsert_profile(req.session_id, redact_data(response.profile.model_dump()))
        self.memory.append_interaction(
            req.session_id,
            redact_data(req.model_dump()),
            redact_data(response.model_dump()),
        )
        self._pending_runs[req.session_id] = (
            req,
            response,
            model_name,
            perception_results,
            knowledge_hints,
            knowledge_hit_ids,
        )

    def submit_feedback(self, session_id: str, feedback: str) -> bool:
        if feedback not in FEEDBACK_OPTIONS:
            raise ValueError(f"unsupported feedback: {feedback}")
        self.memory.append_feedback(session_id, feedback, None)
        pending = self._pending_runs.pop(session_id, None)
        if pending is None:
            return False
        req, response, model_name, *_ = pending
        self.run_logger.log_run(
            req,
            response,
            feedback=feedback,
            model_name=model_name,
        )
        return True

    @staticmethod
    def _response_to_plan_json(response: CareerPlanResponse) -> str:
        """Serialize the effective plan fields for the feedback prompt."""
        return json.dumps(
            {
                "user_facing_advice": response.user_facing_advice,
                "target_roles": response.target_roles,
                "gap_analysis": response.gap_analysis,
                "roadmap_30_90_180": [
                    item.model_dump() for item in response.roadmap_30_90_180
                ],
                "learning_resources": response.learning_resources,
                "next_actions": response.next_actions,
                "risk_flags": response.risk_flags,
                "follow_up_questions": response.follow_up_questions,
                "confidence": response.confidence,
            },
            ensure_ascii=False,
        )

    def adjust_plan(
        self, session_id: str, feedback: str
    ) -> Tuple[Optional[CareerPlanResponse], Optional[str]]:
        """Regenerate the plan for 过短/过于详细 feedback (manual 工作项5).

        - 合适: no regeneration; the original plan is recorded and returned.
        - 过短/过于详细: rebuild the feedback prompt from the saved pipeline
          context, regenerate with the same retry policy, and persist/log the
          adjusted plan. On failure the ORIGINAL plan stays effective (manual
          3.2 保留原结果并记录失败): the error is recorded and returned with
          the original response.
        - No pending run: records the feedback and returns (None, None).

        Returns (effective_response, adjust_error).
        """
        if feedback not in FEEDBACK_OPTIONS:
            raise ValueError(f"unsupported feedback: {feedback}")

        pending = self._pending_runs.get(session_id)
        if pending is None:
            self.memory.append_feedback(session_id, feedback, None)
            return None, None

        req, original, model_name, perception_results, knowledge_hints, knowledge_hit_ids = pending
        if feedback == "合适":
            self.submit_feedback(session_id, feedback)
            return original, None

        started = time.time()
        used_model = req.brain_model or self.settings.brain_default_model
        prompt = build_feedback_prompt(
            original_plan_json=self._response_to_plan_json(original),
            feedback=feedback,
            user_goal=req.user_goal,
            constraints_json=req.constraints.model_dump_json(ensure_ascii=False),
            profile_json=original.profile.model_dump_json(ensure_ascii=False),
            knowledge_hints=knowledge_hints,
        )

        retries = max(0, int(self.settings.brain_retry_times))
        errors: List[str] = []
        attempts = 0
        for i in range(retries + 1):
            attempts = i + 1
            try:
                raw = self.cloud_brain.plan(prompt, model=used_model)
                model_obj = self._extract_json(raw)
                if not model_obj:
                    raise BrainResponseError("反馈调整模型未返回有效 JSON")
                if not self._valid_roadmap(model_obj.get('roadmap_30_90_180')):
                    raise BrainResponseError("反馈调整模型缺少完整的 30/90/180 天路线")
                adjusted = self._compose_from_model_obj(
                    req,
                    original.intent,
                    original.profile,
                    perception_results,
                    knowledge_hints,
                    knowledge_hit_ids,
                    model_obj,
                    served_by='cloud_brain',
                    retry_count=i,
                    model_name=used_model,
                )
                adjusted.latency_ms = int((time.time() - started) * 1000)
                adjusted = self._sanitize_for_user(adjusted, req.debug_trace)
                self._persist_response(
                    req, adjusted, used_model, perception_results, knowledge_hints, knowledge_hit_ids
                )
                self.run_logger.log_run(
                    req,
                    adjusted,
                    feedback=feedback,
                    model_name=used_model,
                    feedback_adjusted=True,
                )
                self._pending_runs.pop(session_id, None)
                return adjusted, None
            except BrainClientError as exc:
                errors.append(exc.code)
                if not exc.retryable:
                    break
            except Exception as exc:
                errors.append(type(exc).__name__)

        error_text = "; ".join(errors) or "feedback adjustment failed"
        self.memory.append_feedback(session_id, feedback, None)
        self.run_logger.log_run(
            req,
            original,
            feedback=feedback,
            model_name=model_name,
            error=error_text,
        )
        self._pending_runs.pop(session_id, None)
        return original, error_text

    def _get_image_agent(self) -> ImagePerceptionAgent:
        if self.image_agent is None:
            try:
                self.image_agent = ImagePerceptionAgent(self.image_model_path)
            except Exception as exc:
                raise RuntimeError(
                    "图像功能不可用，请安装 torch、transformers、qwen-vl-utils 并准备 Qwen3-VL 模型。"
                ) from exc
        return self.image_agent

    def _get_audio_agent(self) -> AudioPerceptionAgent:
        if self.audio_agent is None:
            self.audio_agent = AudioPerceptionAgent()
        return self.audio_agent

    def _get_video_agent(self) -> VideoPerceptionAgent:
        if self.video_agent is None:
            self.video_agent = VideoPerceptionAgent(image_model_path=self.image_model_path)
        return self.video_agent

    @staticmethod
    def detect_intent(query: str) -> str:
        q = query.lower()
        if any(k in q for k in ['复盘', '总结', '回顾', 'review']):
            return 'review'
        if any(k in q for k in ['诊断', '差距', '短板']):
            return 'diagnosis'
        if any(k in q for k in ['规划', '路线', '计划', '转行', 'offer']):
            return 'planning'
        return 'qa'

    @staticmethod
    def _normalize_list(items: List[Any], limit: int = 6) -> List[str]:
        out = []
        seen = set()
        for x in items:
            s = str(x).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _extract_json(text: str):
        text = (text or '').strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        a = text.find('{')
        b = text.rfind('}')
        if a >= 0 and b > a:
            try:
                return json.loads(text[a:b+1])
            except Exception:
                return None
        return None

    @staticmethod
    def _valid_roadmap(items: Any) -> bool:
        if not isinstance(items, list) or len(items) != 3:
            return False
        if not all(
            isinstance(item, dict)
            and isinstance(item.get('objective'), str)
            and item['objective'].strip()
            for item in items
        ):
            return False
        return {item.get('period') for item in items} == {'30d', '90d', '180d'}

    def _collect_perception(self, req: TaskRequest) -> List[PerceptionResult]:
        results: List[PerceptionResult] = []
        text_blob = '\n'.join([req.user_goal.strip(), req.text_input.strip()]).strip()
        if text_blob:
            results.append(self.text_agent.perceive(text_blob))

        for path in req.document_paths:
            results.append(self.document_agent.perceive(path))
        for path in req.audio_paths:
            results.append(self._get_audio_agent().perceive(path))
        for path in req.image_paths:
            results.append(
                self._get_image_agent().perceive(
                    path,
                    user_goal=req.user_goal,
                    user_text=req.text_input,
                )
            )
        for path in req.video_paths:
            results.append(self._get_video_agent().perceive(path))
        return results

    def _build_profile(self, req: TaskRequest, perception_results: List[PerceptionResult]) -> UserProfile:
        memory_profile = self.memory.get_profile(req.session_id)
        facts: List[str] = []
        for p in perception_results:
            facts.extend(p.facts)

        interests = [x for x in facts if any(k in x for k in ['兴趣', '喜欢', '方向', '岗位'])]
        strengths = [x for x in facts if any(k in x for k in ['会', '熟悉', '掌握', '经验', '项目'])]
        weaknesses = [x for x in facts if any(k in x for k in ['缺乏', '不足', '短板', '薄弱'])]

        profile = UserProfile(
            strengths=self._normalize_list(memory_profile.get('strengths', []) + strengths),
            weaknesses=self._normalize_list(memory_profile.get('weaknesses', []) + weaknesses),
            interests=self._normalize_list(memory_profile.get('interests', []) + interests),
            current_stage=memory_profile.get('current_stage', ''),
            constraints=req.constraints,
        )
        if req.follow_up_answers:
            profile = apply_answers_to_profile(profile, req.follow_up_answers)
        return profile

    def _build_planning_prompt(
        self,
        req: TaskRequest,
        intent: str,
        profile: UserProfile,
        perception_results: List[PerceptionResult],
        knowledge_hints: List[str],
    ) -> str:
        perception_text = MultiModalFusion.fuse(perception_results)

        return build_planning_prompt(
            user_goal=req.user_goal,
            text_input=req.text_input,
            intent=intent,
            constraints_json=req.constraints.model_dump_json(ensure_ascii=False),
            profile_json=profile.model_dump_json(ensure_ascii=False),
            perception_text=perception_text,
            knowledge_hints=knowledge_hints,
        )

    def _retrieve_knowledge(self, req: TaskRequest, query: str) -> Tuple[List[str], List[str]]:
        if not req.use_knowledge:
            return [], []

        hits = self.knowledge.retrieve(query, top_k=4)
        hints = [
            (
                f"{hit['role']} | 核心技能: {hit['skills']} | "
                f"薪资参考: {hit['salary_hint']}"
            )
            for hit in hits
        ]
        hit_ids = [hit["item_id"] for hit in hits if hit.get("item_id")]
        return hints, hit_ids

    def _planner_fallback(
        self,
        req: TaskRequest,
        intent: str,
        profile: UserProfile,
        perception_results: List[PerceptionResult],
        knowledge_hints: List[str],
        knowledge_hit_ids: List[str],
        retry_count: int = 0,
    ) -> CareerPlanResponse:
        missing = []
        for p in perception_results:
            missing.extend(p.missing_info)

        target_roles = self._normalize_list([h.split('|')[0].strip() for h in knowledge_hints], limit=3)
        gap = self._normalize_list(
            profile.weaknesses + [
                '缺少可量化项目成果',
                '岗位能力与业务场景连接不足',
                '简历叙事与岗位关键词匹配度不高',
            ],
            limit=8,
        )
        roadmap = [
            Milestone(
                period='30d',
                objective='完成方向收敛与能力盘点',
                deliverables=['确定1-2个目标岗位', '输出能力差距清单', '重写简历基础版'],
                metrics=['完成2份岗位JD拆解', '每周投入>=8小时'],
            ),
            Milestone(
                period='90d',
                objective='建立可投递的项目与作品证据',
                deliverables=['完成1-2个岗位相关项目', '完善项目文档与复盘', '进行模拟面试'],
                metrics=['至少1个公开作品链接', '完成6次模拟面试问答'],
            ),
            Milestone(
                period='180d',
                objective='规模化投递与面试迭代',
                deliverables=['形成A/B版简历与自我介绍', '建立投递-面试-复盘看板', '迭代短板课程计划'],
                metrics=['累计60次有效投递', '获取>=6次面试机会'],
            ),
        ]

        return CareerPlanResponse(
            session_id=req.session_id,
            intent=intent,  # type: ignore[arg-type]
            profile=profile,
            user_facing_advice='''建议先收敛到1-2个目标岗位，再用30/90/180天路线推进。\n\n优先把项目证据补齐，再进入规模化投递。''',
            target_roles=target_roles or ['数据分析师', '测试开发工程师'],
            gap_analysis=gap,
            roadmap_30_90_180=roadmap,
            learning_resources=self._normalize_list(knowledge_hints + ['LeetCode', 'Kaggle', '岗位JD反向拆解模板'], 10),
            next_actions=self._normalize_list([
                '今天完成2个目标岗位JD拆解',
                '本周输出一版简历并找1位同伴评审',
                '设定每周固定学习与项目时段',
            ], 10),
            risk_flags=self._normalize_list(['目标岗位过多导致精力分散', '学习输入多输出少', '缺少真实反馈闭环'], 8),
            follow_up_questions=self._normalize_list(missing + ['你每周稳定可投入多少小时？', '你更偏好技术深度路线还是业务综合路线？'], 8),
            confidence=0.55,
            perception_results=perception_results,
            knowledge_hits=knowledge_hints,
            knowledge_hit_ids=knowledge_hit_ids,
            model_trace=[
                'text-perception: rule-based',
                f'image-perception: {self.image_model_path}',
                'brain: local-fallback',
            ],
            served_by='local_fallback',
            retry_count=retry_count,
        )

    def _sanitize_for_user(self, response: CareerPlanResponse, debug_trace: bool) -> CareerPlanResponse:
        if debug_trace:
            return response
        for p in response.perception_results:
            p.raw_output = ''
        return response

    def _compose_from_model_obj(
        self,
        req: TaskRequest,
        intent: str,
        profile: UserProfile,
        perception_results: List[PerceptionResult],
        knowledge_hints: List[str],
        knowledge_hit_ids: List[str],
        model_obj: Dict[str, Any],
        served_by: str,
        retry_count: int,
        model_name: str,
    ) -> CareerPlanResponse:
        gap = model_obj.get('gap_analysis', [])
        if not isinstance(gap, list):
            gap = []
        gap = [x for x in gap if isinstance(x, str) and '{' not in x]

        risk_flags = model_obj.get('risk_flags', [])
        if not isinstance(risk_flags, list):
            risk_flags = []
        risk_flags = [x for x in risk_flags if isinstance(x, str) and x.strip() and x.strip() != '无风险']

        return CareerPlanResponse(
            session_id=req.session_id,
            intent=intent,  # type: ignore[arg-type]
            profile=profile,
            user_facing_advice=str(model_obj.get('user_facing_advice', '')).strip(),
            target_roles=self._normalize_list(model_obj.get('target_roles', []), 4),
            gap_analysis=self._normalize_list(gap, 8),
            roadmap_30_90_180=[
                Milestone(
                    period=item.get('period', '30d'),
                    objective=item.get('objective', ''),
                    deliverables=self._normalize_list(item.get('deliverables', []), 6),
                    metrics=self._normalize_list(item.get('metrics', []), 5),
                )
                for item in model_obj.get('roadmap_30_90_180', [])
                if isinstance(item, dict)
            ][:3],
            learning_resources=self._normalize_list(model_obj.get('learning_resources', []), 10),
            next_actions=self._normalize_list(model_obj.get('next_actions', []), 10),
            risk_flags=self._normalize_list(risk_flags, 8),
            follow_up_questions=self._normalize_list(model_obj.get('follow_up_questions', []), 8),
            confidence=max(0.35, min(0.95, float(model_obj.get('confidence', 0.6)))),
            perception_results=perception_results,
            knowledge_hits=knowledge_hints,
            knowledge_hit_ids=knowledge_hit_ids,
            model_trace=[
                'text-perception: rule-based',
                f'image-perception: {self.image_model_path}',
                f'brain: {model_name}',
            ],
            served_by='cloud_brain' if served_by == 'cloud_brain' else 'local_fallback',
            retry_count=retry_count,
        )

    def _run_core(self, req: TaskRequest) -> Tuple[CareerPlanResponse, int]:
        t0 = time.time()
        query = f"{req.user_goal}\n{req.text_input}".strip()
        intent = self.detect_intent(query)
        perception_results = self._collect_perception(req)
        profile = self._build_profile(req, perception_results)
        knowledge_hints, knowledge_hit_ids = self._retrieve_knowledge(req, query)

        if req.planner_mode == "template":
            resp = self._planner_fallback(
                req,
                intent,
                profile,
                perception_results,
                knowledge_hints,
                knowledge_hit_ids,
            )
            resp.latency_ms = int((time.time() - t0) * 1000)
            resp = self._sanitize_for_user(resp, req.debug_trace)
            self._persist_response(req, resp, "local-template", perception_results, knowledge_hints, knowledge_hit_ids)
            return resp, 0

        prompt = self._build_planning_prompt(req, intent, profile, perception_results, knowledge_hints)

        retries = max(0, int(self.settings.brain_retry_times))
        errors: List[str] = []
        attempts = 0
        used_model = req.brain_model or self.settings.brain_default_model
        for i in range(retries + 1):
            attempts = i + 1
            try:
                raw = self.cloud_brain.plan(prompt, model=used_model)
                model_obj = self._extract_json(raw)
                if not model_obj:
                    raise BrainResponseError("规划模型未返回有效 JSON")
                if not self._valid_roadmap(model_obj.get('roadmap_30_90_180')):
                    raise BrainResponseError("规划模型缺少完整的 30/90/180 天路线")
                resp = self._compose_from_model_obj(
                    req,
                    intent,
                    profile,
                    perception_results,
                    knowledge_hints,
                    knowledge_hit_ids,
                    model_obj,
                    served_by='cloud_brain',
                    retry_count=i,
                    model_name=used_model,
                )
                resp.latency_ms = int((time.time() - t0) * 1000)
                resp = self._sanitize_for_user(resp, req.debug_trace)
                self._persist_response(req, resp, used_model, perception_results, knowledge_hints, knowledge_hit_ids)
                return resp, i
            except BrainClientError as exc:
                errors.append(exc.code)
                if not exc.retryable:
                    break
            except Exception as exc:
                errors.append(type(exc).__name__)

        resp = self._planner_fallback(
            req,
            intent,
            profile,
            perception_results,
            knowledge_hints,
            knowledge_hit_ids,
            retry_count=max(0, attempts - 1),
        )
        resp.latency_ms = int((time.time() - t0) * 1000)
        if errors and req.debug_trace:
            resp.follow_up_questions = self._normalize_list(
                resp.follow_up_questions + [f'cloud_error: {errors[-1]}'],
                8,
            )
        resp = self._sanitize_for_user(resp, req.debug_trace)
        self._persist_response(req, resp, "local-template", perception_results, knowledge_hints, knowledge_hit_ids)
        return resp, resp.retry_count

    def run(self, req: TaskRequest) -> CareerPlanResponse:
        resp, _ = self._run_core(req)
        return resp

    def run_stream(self, req: TaskRequest) -> Generator[Dict[str, Any], None, None]:
        start = time.time()
        yield {'event': 'stage_start', 'data': {'stage': 'input_understanding'}}

        query = f"{req.user_goal}\n{req.text_input}".strip()
        intent = self.detect_intent(query)
        yield {'event': 'stage_end', 'data': {'stage': 'input_understanding', 'intent': intent}}

        yield {'event': 'stage_start', 'data': {'stage': 'perception', 'model': 'rule-based-text'}}
        perception_results = self._collect_perception(req)
        yield {'event': 'stage_end', 'data': {'stage': 'perception', 'count': len(perception_results)}}

        profile = self._build_profile(req, perception_results)
        knowledge_hints, knowledge_hit_ids = self._retrieve_knowledge(req, query)

        if req.planner_mode == "template":
            fallback = self._planner_fallback(
                req,
                intent,
                profile,
                perception_results,
                knowledge_hints,
                knowledge_hit_ids,
            )
            fallback.latency_ms = int((time.time() - start) * 1000)
            fallback = self._sanitize_for_user(fallback, req.debug_trace)
            self._persist_response(req, fallback, "local-template", perception_results, knowledge_hints, knowledge_hit_ids)
            yield {'event': 'stage_end', 'data': {'stage': 'brain_planning', 'served_by': 'local_fallback'}}
            yield {'event': 'final_result', 'data': fallback.model_dump()}
            return

        prompt = self._build_planning_prompt(req, intent, profile, perception_results, knowledge_hints)
        used_model = req.brain_model or self.settings.brain_default_model

        yield {'event': 'stage_start', 'data': {'stage': 'brain_planning', 'model': used_model}}

        retries = max(0, int(self.settings.brain_retry_times))
        attempts = 0
        for i in range(retries + 1):
            attempts = i + 1
            try:
                token_buf = ''
                for token in self.cloud_brain.plan_stream(prompt, model=used_model):
                    token_buf += token
                    yield {'event': 'token', 'data': {'stage': 'brain_planning', 'token': token}}

                model_obj = self._extract_json(token_buf)
                if not model_obj:
                    raise BrainResponseError("规划模型未返回有效 JSON")
                if not self._valid_roadmap(model_obj.get('roadmap_30_90_180')):
                    raise BrainResponseError("规划模型缺少完整的 30/90/180 天路线")
                resp = self._compose_from_model_obj(
                    req, intent, profile, perception_results, knowledge_hints, knowledge_hit_ids, model_obj,
                    served_by='cloud_brain', retry_count=i, model_name=used_model,
                )
                resp.latency_ms = int((time.time() - start) * 1000)
                resp = self._sanitize_for_user(resp, req.debug_trace)
                self._persist_response(req, resp, used_model, perception_results, knowledge_hints, knowledge_hit_ids)
                yield {'event': 'stage_end', 'data': {'stage': 'brain_planning', 'retry': i}}
                yield {'event': 'final_result', 'data': resp.model_dump()}
                return
            except BrainClientError as exc:
                yield {
                    'event': 'stage_progress',
                    'data': {'stage': 'brain_planning', 'retry': i, 'error': exc.code},
                }
                if not exc.retryable:
                    break
            except Exception as exc:
                yield {
                    'event': 'stage_progress',
                    'data': {
                        'stage': 'brain_planning',
                        'retry': i,
                        'error': type(exc).__name__,
                    },
                }

        fallback = self._planner_fallback(
            req,
            intent,
            profile,
            perception_results,
            knowledge_hints,
            knowledge_hit_ids,
            retry_count=max(0, attempts - 1),
        )
        fallback.latency_ms = int((time.time() - start) * 1000)
        fallback = self._sanitize_for_user(fallback, req.debug_trace)
        self._persist_response(req, fallback, "local-template", perception_results, knowledge_hints, knowledge_hit_ids)
        yield {'event': 'stage_end', 'data': {'stage': 'brain_planning', 'served_by': 'local_fallback'}}
        yield {'event': 'final_result', 'data': fallback.model_dump()}
