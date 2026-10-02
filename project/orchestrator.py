from __future__ import annotations

import json
import time
import hashlib
import uuid
from dataclasses import dataclass
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
from project.utils.fusion import MultiModalFusion
from project.core.guidance import GuidanceModel, collect_evidence, merge_decision, run_guidance
from project.core.intake import add_evidence, field_value, initialize_fields
from project.core.local_guidance import LocalGuidanceModel
from project.core.schemas import GuidanceDecision, GuidanceObservation
from project.core.local_feedback import LocalFeedbackModel
from project.core.output_adaptation import FeedbackModel, adapt_output, original_version
from project.core.schemas import FeedbackAdaptationResult


@dataclass
class PreparedCareerInput:
    request_json: str
    perception_results: List[PerceptionResult]
    profile: UserProfile


class CareerOrchestrator:
    def __init__(
        self,
        image_model_path: Optional[str] = None,
        db_path: str = './data/session_memory.db',
        brain_client: Optional[BrainClient] = None,
        run_logger: Optional[JsonlRunLogger] = None,
        guidance_model: Optional[GuidanceModel] = None,
        feedback_model: Optional[FeedbackModel] = None,
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
        self._pending_runs: Dict[str, Tuple[TaskRequest, CareerPlanResponse, str]] = {}

        self.image_model_path = image_model_path or self.settings.vision_model_path
        self.guidance_model = guidance_model or LocalGuidanceModel(
            self.settings.guidance_model_path, self.settings.local_model_device,
        )
        self.feedback_model = feedback_model or LocalFeedbackModel(
            self.settings.feedback_model_path or self.settings.guidance_model_path,
            self.settings.local_model_device,
        )

    def _persist_response(
        self,
        req: TaskRequest,
        response: CareerPlanResponse,
        model_name: str,
    ) -> None:
        response.plan_id = uuid.uuid4().hex
        response.output_version = original_version(response)
        self.memory.save_output_version(req.session_id, redact_data(response.output_version.model_dump()))
        self.memory.upsert_profile(req.session_id, redact_data(response.profile.model_dump()))
        self.memory.append_interaction(
            req.session_id,
            redact_data(req.model_dump()),
            redact_data(response.model_dump()),
        )
        self._pending_runs[req.session_id] = (req, response, model_name)

    def submit_feedback(self, session_id: str, feedback: str) -> bool:
        if feedback not in FEEDBACK_OPTIONS:
            raise ValueError(f"unsupported feedback: {feedback}")
        self.memory.append_feedback(session_id, feedback, None)
        pending = self._pending_runs.pop(session_id, None)
        if pending is None:
            return False
        req, response, model_name = pending
        self.run_logger.log_run(
            req,
            response,
            feedback=feedback,
            model_name=model_name,
        )
        return True

    def adapt_feedback(
        self, session_id: str, feedback: str, *, request_id: Optional[str] = None,
        plan_id: Optional[str] = None,
    ) -> FeedbackAdaptationResult:
        """One adaptation per active plan; identical requests replay durably."""
        if feedback not in FEEDBACK_OPTIONS:
            raise ValueError(f"unsupported feedback: {feedback}")
        pending = self._pending_runs.get(session_id)
        latest = self.memory.get_latest_output_version(session_id)
        active_plan = pending[1].plan_id if pending else (latest or {}).get("plan_id")
        plan_id = plan_id or active_plan
        if not plan_id:
            raise ValueError("No planning output is available for this session")
        if request_id is None:
            request_id = hashlib.sha256(f"{session_id}\n{plan_id}\n{feedback}".encode()).hexdigest()
        cached = self.memory.get_feedback_result(request_id)
        if cached:
            result = FeedbackAdaptationResult.model_validate(cached)
            if (result.session_id, result.plan_id, result.feedback) != (session_id, plan_id, feedback):
                raise ValueError("Feedback request ID belongs to different input")
            if pending and pending[1].plan_id == plan_id:
                req, response, model_name = pending
                self.run_logger.log_run(req, response, feedback=feedback, model_name=model_name, adaptation=result)
                self._pending_runs.pop(session_id)
            else:
                self.run_logger.log_feedback_replay(result, self.memory.get_output_versions(session_id, plan_id))
            return result
        if not pending or pending[1].plan_id != plan_id:
            raise ValueError("The plan is no longer active; only a saved feedback request can be replayed")
        req, response, model_name = pending
        result = adapt_output(response, feedback, request_id, self.feedback_model)
        self.memory.save_feedback_result(redact_data(result.model_dump()))
        self.run_logger.log_run(req, response, feedback=feedback, model_name=model_name, adaptation=result)
        self._pending_runs.pop(session_id)
        return result

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
            self.audio_agent = AudioPerceptionAgent(
                model_path=self.settings.audio_model_path,
                device=self.settings.local_model_device,
            )
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
        profile = UserProfile.model_validate(self.memory.get_profile(req.session_id))
        initialize_fields(profile)
        facts = [fact for result in perception_results for fact in result.facts]
        profile.strengths = self._normalize_list(profile.strengths + [
            fact for fact in facts if any(k in fact for k in ['会', '熟悉', '掌握', '经验', '项目'])
        ])
        profile.weaknesses = self._normalize_list(profile.weaknesses + [
            fact for fact in facts if any(k in fact for k in ['缺乏', '不足', '短板', '薄弱'])
        ])
        supplied = req.constraints.model_dump(exclude_none=True)
        observations = []
        for key, field in [('education_level', 'education'), ('time_budget_hours_per_week', 'time_budget')]:
            if key in supplied:
                value = str(supplied[key])
                ref = add_evidence(profile, f'cli_constraint:{field}', value)
                observations.append(GuidanceObservation(field=field, value=value, evidence_ids=[ref]))
        preference = '、'.join(filter(None, [supplied.get('city'), *supplied.get('preferred_industries', [])]))
        if preference:
            ref = add_evidence(profile, 'cli_constraint:preference', preference)
            observations.append(GuidanceObservation(field='preference', value=preference, evidence_ids=[ref]))
        profile = merge_decision(profile, GuidanceDecision(observations=observations))
        if 'financial_budget_cny' in supplied:
            profile.constraints.financial_budget_cny = supplied['financial_budget_cny']
        if preference and profile.guidance.fields['preference'].status == 'known':
            if supplied.get('city'):
                profile.constraints.city = supplied['city']
            if supplied.get('preferred_industries'):
                profile.constraints.preferred_industries = supplied['preferred_industries']
        if req.follow_up_answers:
            profile = apply_answers_to_profile(profile, req.follow_up_answers)
        return profile

    def prepare(self, req: TaskRequest, *, input_fn=input, output_fn=print) -> PreparedCareerInput:
        results = self._collect_perception(req)
        profile = self._build_profile(req, results)
        profile = run_guidance(
            req, profile, results, self.guidance_model,
            max_rounds=self.settings.guidance_max_rounds,
            input_fn=input_fn, output_fn=output_fn,
        )
        req.guidance = profile.guidance.model_copy(deep=True)
        return PreparedCareerInput(req.model_dump_json(), results, profile)

    def _planning_context(self, req: TaskRequest, prepared: Optional[PreparedCareerInput]):
        if prepared is None:
            results = self._collect_perception(req)
            profile = self._build_profile(req, results)
            collect_evidence(profile, req, results)
            profile.guidance.stop_reason = 'skipped' if req.skip_follow_up else 'not_requested'
            profile.guidance.model_used = False
            req.guidance = profile.guidance.model_copy(deep=True)
        else:
            if prepared.request_json != req.model_dump_json():
                raise ValueError('Prepared input does not match the planning request')
            results, profile = prepared.perception_results, prepared.profile
        query = '\n'.join(filter(None, [
            req.user_goal, req.text_input,
            *[field_value(profile, field) for field in ('target_role', 'skills', 'interests', 'major')
              if profile.guidance.fields[field].status == 'known'],
        ]))
        return results, profile, query

    def _build_planning_prompt(
        self,
        req: TaskRequest,
        intent: str,
        profile: UserProfile,
        perception_results: List[PerceptionResult],
        knowledge_hints: List[str],
    ) -> str:
        perception_text = MultiModalFusion.fuse(perception_results)

        return f"""
你是职业规划总控代理。请只输出严格 JSON，不要输出其他文本。
JSON schema:
{{
  "user_facing_advice": "面向用户的自然语言建议（分段，行动导向）",
  "target_roles": ["岗位1", "岗位2"],
  "gap_analysis": ["差距1", "差距2"],
  "roadmap_30_90_180": [
    {{"period":"30d","objective":"...","deliverables":["..."],"metrics":["..."]}},
    {{"period":"90d","objective":"...","deliverables":["..."],"metrics":["..."]}},
    {{"period":"180d","objective":"...","deliverables":["..."],"metrics":["..."]}}
  ],
  "learning_resources": ["资源1"],
  "next_actions": ["下一步1"],
  "risk_flags": ["风险1"],
  "follow_up_questions": ["追问1"],
  "confidence": 0.0
}}

用户目标: {req.user_goal}
用户文本: {req.text_input}
意图: {intent}
约束: {profile.constraints.model_dump_json(ensure_ascii=False)}
画像中的 guidance 记录回答及证据。unknown/refused/missing/conflict 都不是已确认事实；
explicit_none 表示用户明确没有。不要推断未解决项已补齐，规划须说明其限制。
用户画像: {profile.model_dump_json(ensure_ascii=False)}
多模态感知结构化结果:
{perception_text}
知识库提示:
{json.dumps(knowledge_hints, ensure_ascii=False)}
"""

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

    def _run_core(self, req: TaskRequest, prepared: Optional[PreparedCareerInput] = None) -> Tuple[CareerPlanResponse, int]:
        t0 = time.time()
        query = f"{req.user_goal}\n{req.text_input}".strip()
        intent = self.detect_intent(query)
        perception_results, profile, query = self._planning_context(req, prepared)
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
            self._persist_response(req, resp, "local-template")
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
                self._persist_response(req, resp, used_model)
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
        self._persist_response(req, resp, "local-template")
        return resp, resp.retry_count

    def run(self, req: TaskRequest, *, prepared: Optional[PreparedCareerInput] = None) -> CareerPlanResponse:
        resp, _ = self._run_core(req, prepared)
        return resp

    def run_stream(self, req: TaskRequest, *, prepared: Optional[PreparedCareerInput] = None) -> Generator[Dict[str, Any], None, None]:
        start = time.time()
        yield {'event': 'stage_start', 'data': {'stage': 'input_understanding'}}

        query = f"{req.user_goal}\n{req.text_input}".strip()
        intent = self.detect_intent(query)
        yield {'event': 'stage_end', 'data': {'stage': 'input_understanding', 'intent': intent}}

        yield {'event': 'stage_start', 'data': {'stage': 'perception', 'model': 'rule-based-text'}}
        perception_results, profile, query = self._planning_context(req, prepared)
        yield {'event': 'stage_end', 'data': {'stage': 'perception', 'count': len(perception_results)}}

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
            self._persist_response(req, fallback, "local-template")
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
                self._persist_response(req, resp, used_model)
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
        self._persist_response(req, fallback, "local-template")
        yield {'event': 'stage_end', 'data': {'stage': 'brain_planning', 'served_by': 'local_fallback'}}
        yield {'event': 'final_result', 'data': fallback.model_dump()}
