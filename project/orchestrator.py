from __future__ import annotations

import json
import time
from typing import Any, Dict, Generator, List, Tuple

try:
    from .core.schemas import CareerPlanResponse, Milestone, PerceptionResult, TaskRequest, UserProfile
    from .core.brain_client import DeepSeekBrainClient
    from .core.career_knowledge import CareerKnowledgeBase
    from .agents.perception import AudioPerceptionAgent, DocumentPerceptionAgent, ImagePerceptionAgent, TextPerceptionAgent, VideoPerceptionAgent
    from .core.session_memory import SessionMemory
    from .core.settings import get_settings
except ImportError:
    from project.core.schemas import CareerPlanResponse, Milestone, PerceptionResult, TaskRequest, UserProfile
    from project.core.brain_client import DeepSeekBrainClient
    from project.core.career_knowledge import CareerKnowledgeBase
    from project.agents.perception import AudioPerceptionAgent, DocumentPerceptionAgent, ImagePerceptionAgent, TextPerceptionAgent, VideoPerceptionAgent
    from project.core.session_memory import SessionMemory
    from project.core.settings import get_settings
from project.utils.fusion import MultiModalFusion


class CareerOrchestrator:
    def __init__(
        self,
        image_model_path: str = './models/Qwen3-VL-2B-Instruct',
        db_path: str = './data/session_memory.db',
    ):
        self.settings = get_settings()
        self.memory = SessionMemory(db_path=db_path)
        self.knowledge = CareerKnowledgeBase()
        self.text_agent = TextPerceptionAgent()
        self.image_agent = ImagePerceptionAgent(image_model_path)
        self.document_agent = DocumentPerceptionAgent()
        self.audio_agent = AudioPerceptionAgent()
        self.video_agent = VideoPerceptionAgent(image_model_path=image_model_path)

        self.cloud_brain = DeepSeekBrainClient()

        self.image_model_path = image_model_path

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
        if not isinstance(items, list) or len(items) == 0:
            return False
        return all(isinstance(x, dict) and 'period' in x and 'objective' in x for x in items)

    def _collect_perception(self, req: TaskRequest) -> List[PerceptionResult]:
        results: List[PerceptionResult] = []
        text_blob = '\n'.join([req.user_goal.strip(), req.text_input.strip()]).strip()
        if text_blob:
            results.append(self.text_agent.perceive(text_blob))

        for path in req.document_paths:
            results.append(self.document_agent.perceive(path))
        for path in req.audio_paths:
            results.append(self.audio_agent.perceive(path))
        for path in req.image_paths:
            results.append(self.image_agent.perceive(path, user_goal=req.user_goal, user_text=req.text_input))
        for path in req.video_paths:
            results.append(self.video_agent.perceive(path))
        return results

    def _build_profile(self, req: TaskRequest, perception_results: List[PerceptionResult]) -> UserProfile:
        memory_profile = self.memory.get_profile(req.session_id)
        facts: List[str] = []
        for p in perception_results:
            facts.extend(p.facts)

        interests = [x for x in facts if any(k in x for k in ['兴趣', '喜欢', '方向', '岗位'])]
        strengths = [x for x in facts if any(k in x for k in ['会', '熟悉', '掌握', '经验', '项目'])]
        weaknesses = [x for x in facts if any(k in x for k in ['缺乏', '不足', '短板', '薄弱'])]

        return UserProfile(
            strengths=self._normalize_list(memory_profile.get('strengths', []) + strengths),
            weaknesses=self._normalize_list(memory_profile.get('weaknesses', []) + weaknesses),
            interests=self._normalize_list(memory_profile.get('interests', []) + interests),
            current_stage=memory_profile.get('current_stage', ''),
            constraints=req.constraints,
        )

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
约束: {req.constraints.model_dump_json(ensure_ascii=False)}
用户画像: {profile.model_dump_json(ensure_ascii=False)}
多模态感知结构化结果:
{perception_text}
知识库提示:
{json.dumps(knowledge_hints, ensure_ascii=False)}
"""

    def _planner_fallback(
        self,
        req: TaskRequest,
        intent: str,
        profile: UserProfile,
        perception_results: List[PerceptionResult],
        knowledge_hints: List[str],
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
            model_trace=[
                'text-perception: rule-based',
                f'image-perception: {self.image_model_path}',
                'brain: local-fallback',
            ],
            served_by='local_fallback',
            retry_count=self.settings.brain_retry_times,
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
        knowledge_hints = self.knowledge.to_hints(query, top_k=4)
        prompt = self._build_planning_prompt(req, intent, profile, perception_results, knowledge_hints)

        retries = max(0, int(self.settings.brain_retry_times))
        errors = []
        used_model = req.brain_model or self.settings.brain_default_model
        for i in range(retries + 1):
            try:
                raw = self.cloud_brain.plan(prompt, model=used_model)
                model_obj = self._extract_json(raw)
                if model_obj and self._valid_roadmap(model_obj.get('roadmap_30_90_180')):
                    resp = self._compose_from_model_obj(
                        req,
                        intent,
                        profile,
                        perception_results,
                        knowledge_hints,
                        model_obj,
                        served_by='cloud_brain',
                        retry_count=i,
                        model_name=used_model,
                    )
                    resp.latency_ms = int((time.time() - t0) * 1000)
                    self.memory.upsert_profile(req.session_id, resp.profile.model_dump())
                    self.memory.append_interaction(req.session_id, req.model_dump(), resp.model_dump())
                    return self._sanitize_for_user(resp, req.debug_trace), i
            except Exception as e:
                errors.append(str(e))

        resp = self._planner_fallback(req, intent, profile, perception_results, knowledge_hints)
        resp.latency_ms = int((time.time() - t0) * 1000)
        if errors and req.debug_trace:
            resp.follow_up_questions = self._normalize_list(resp.follow_up_questions + [f'cloud_error: {errors[-1]}'], 8)
        self.memory.upsert_profile(req.session_id, resp.profile.model_dump())
        self.memory.append_interaction(req.session_id, req.model_dump(), resp.model_dump())
        return self._sanitize_for_user(resp, req.debug_trace), retries

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
        knowledge_hints = self.knowledge.to_hints(query, top_k=4)
        prompt = self._build_planning_prompt(req, intent, profile, perception_results, knowledge_hints)
        used_model = req.brain_model or self.settings.brain_default_model

        yield {'event': 'stage_start', 'data': {'stage': 'brain_planning', 'model': used_model}}

        retries = max(0, int(self.settings.brain_retry_times))
        for i in range(retries + 1):
            try:
                token_buf = ''
                for token in self.cloud_brain.plan_stream(prompt, model=used_model):
                    token_buf += token
                    yield {'event': 'token', 'data': {'stage': 'brain_planning', 'token': token}}

                model_obj = self._extract_json(token_buf)
                if model_obj and self._valid_roadmap(model_obj.get('roadmap_30_90_180')):
                    resp = self._compose_from_model_obj(
                        req, intent, profile, perception_results, knowledge_hints, model_obj,
                        served_by='cloud_brain', retry_count=i, model_name=used_model,
                    )
                    resp.latency_ms = int((time.time() - start) * 1000)
                    resp = self._sanitize_for_user(resp, req.debug_trace)
                    self.memory.upsert_profile(req.session_id, resp.profile.model_dump())
                    self.memory.append_interaction(req.session_id, req.model_dump(), resp.model_dump())
                    yield {'event': 'stage_end', 'data': {'stage': 'brain_planning', 'retry': i}}
                    yield {'event': 'final_result', 'data': resp.model_dump()}
                    return
            except Exception as e:
                yield {'event': 'stage_progress', 'data': {'stage': 'brain_planning', 'retry': i, 'error': str(e)}}

        fallback = self._planner_fallback(req, intent, profile, perception_results, knowledge_hints)
        fallback.latency_ms = int((time.time() - start) * 1000)
        fallback = self._sanitize_for_user(fallback, req.debug_trace)
        self.memory.upsert_profile(req.session_id, fallback.profile.model_dump())
        self.memory.append_interaction(req.session_id, req.model_dump(), fallback.model_dump())
        yield {'event': 'stage_end', 'data': {'stage': 'brain_planning', 'served_by': 'local_fallback'}}
        yield {'event': 'final_result', 'data': fallback.model_dump()}
