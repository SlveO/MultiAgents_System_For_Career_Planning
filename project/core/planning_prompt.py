"""Versioned planning prompt for the career planner (member B).

Prompt engineering is member B's responsibility, so the template lives here
instead of inside transport (brain_client.py) or orchestration
(orchestrator.py). The lead freezes prompt versions for experiments; bump
PLANNING_PROMPT_VERSION whenever the template changes so run records stay
reproducible.

The prompt demands strict JSON and the manual's required planning parts:
direction (target_roles), gap analysis (gap_analysis), and phased actions
(roadmap_30_90_180). It also requires every career fact to be traceable to
the injected knowledge hints; unsupported claims must be flagged instead of
invented.
"""

from __future__ import annotations

from typing import List

PLANNING_PROMPT_VERSION = "planning-v1"


def build_planning_prompt(
    user_goal: str,
    text_input: str,
    intent: str,
    constraints_json: str,
    profile_json: str,
    perception_text: str,
    knowledge_hints: List[str],
) -> str:
    """Render the planning prompt from already-serialized pipeline inputs.

    String-based on purpose: the module stays free of schema imports, so it
    can be reused by both the DeepSeek MVP path and the local-model
    experiment path without pulling pydantic objects.
    """
    if knowledge_hints:
        knowledge_text = "\n".join(f"- {hint}" for hint in knowledge_hints)
    else:
        knowledge_text = "（空）——知识库未命中，请按要求3处理"

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

要求:
1. 方向(target_roles)、差距(gap_analysis)、阶段行动(roadmap_30_90_180)必须齐全且相互呼应;
2. 规划必须引用"知识库提示"中的事实，岗位与技能描述应能在知识库提示中找到出处，不得虚构知识库未提供的职业事实;
3. 知识库提示为空或与用户目标不匹配时，在 risk_flags 中注明"知识库未命中"，并仅基于用户画像给出保守建议。

用户目标: {user_goal}
用户文本: {text_input}
意图: {intent}
约束: {constraints_json}
用户画像: {profile_json}
多模态感知结构化结果:
{perception_text}
知识库提示:
{knowledge_text}
"""
