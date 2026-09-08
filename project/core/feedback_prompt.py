"""Versioned feedback-adjustment prompt for the career planner (member B).

Implements the manual's three-tier output adjustment (工作项5): the user
reports 过短 (too short), 合适 (suitable), or 过于详细 (too detailed) and the
planner regenerates accordingly. 合适 needs no regeneration and never reaches
this module; 过短 expands the plan and 过于详细 condenses it, both while
keeping the same strict-JSON contract as planning-v1 and forbidding invented
career facts.

The lead freezes prompt versions for experiments; bump FEEDBACK_PROMPT_VERSION
whenever the template changes so run records stay reproducible.
"""

from __future__ import annotations

from typing import Dict, List

from project.core.planning_prompt import (
    PLANNING_JSON_SCHEMA_TEXT,
    PLANNING_REQUIREMENTS_TEXT,
)

FEEDBACK_PROMPT_VERSION = "feedback-v1"

# Per-tier regeneration instructions. The keys are the values of
# project.core.feedback.FEEDBACK_OPTIONS minus 合适.
TIER_INSTRUCTIONS: Dict[str, str] = {
    "过短": """用户认为当前规划内容过短。请在原规划基础上扩充，具体到每周行动:
- user_facing_advice 分段且行动导向，每段给出可执行做法，篇幅至少为原规划的 1.5 倍;
- gap_analysis 不少于 4 条，每条包含现象与原因;
- roadmap_30_90_180 每阶段 deliverables 不少于 3 条、metrics 不少于 2 条;
- learning_resources 不少于 5 条，next_actions 不少于 5 条;
- 保持 30/90/180 三阶段结构与原有职业方向，不得因扩充而虚构知识库未提供的事实。""",
    "过于详细": """用户认为当前规划过于详细。请精简，只保留最关键内容:
- user_facing_advice 不超过 200 字，只写核心结论与最关键的下一步;
- target_roles 不超过 2 个，gap_analysis 不超过 3 条且每条一句话;
- roadmap_30_90_180 每阶段 deliverables 不超过 2 条、metrics 不超过 2 条;
- learning_resources 不超过 3 条，next_actions 不超过 3 条;
- 保持 30/90/180 三阶段结构与原有职业方向，精简时不得丢失任何一阶段。""",
}


def build_feedback_prompt(
    original_plan_json: str,
    feedback: str,
    user_goal: str,
    constraints_json: str,
    profile_json: str,
    knowledge_hints: List[str],
) -> str:
    """Render the regeneration prompt for 过短 or 过于详细 feedback.

    String-based on purpose, mirroring build_planning_prompt, so the module
    stays free of schema imports. Raises ValueError for unsupported feedback
    values; callers must route 合适 through the no-regeneration path.
    """
    if feedback not in TIER_INSTRUCTIONS:
        raise ValueError(f"unsupported feedback tier: {feedback}")
    if knowledge_hints:
        knowledge_text = "\n".join(f"- {hint}" for hint in knowledge_hints)
    else:
        knowledge_text = "（空）——知识库未命中，请按要求处理"

    return f"""
你是职业规划总控代理。用户对上一版规划给出了档位反馈，请按反馈档位重新生成规划。
请只输出严格 JSON，不要输出其他文本。
JSON schema:
{PLANNING_JSON_SCHEMA_TEXT}

要求:
{PLANNING_REQUIREMENTS_TEXT}
4. 调整必须基于下方"上一版规划"，保持职业方向与已给事实一致，不得无中生有;
5. 档位指令（反馈档位）是本轮输出的唯一修改依据，超出档位指令的内容保持与原规划一致。

反馈档位: {feedback}
档位指令:
{TIER_INSTRUCTIONS[feedback]}

用户目标: {user_goal}
约束: {constraints_json}
用户画像: {profile_json}
知识库提示:
{knowledge_text}
上一版规划 JSON:
{original_plan_json}
"""
