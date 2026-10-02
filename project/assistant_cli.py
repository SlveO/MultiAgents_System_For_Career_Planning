from __future__ import annotations

import argparse
import json
import sys
from typing import Callable, Sequence

try:
    from .core.intake import FIELD_PATHS
    from .core.feedback import collect_feedback
    from .core.schemas import TaskRequest, UserConstraints
    from .orchestrator import CareerOrchestrator
except ImportError:
    from project.core.intake import FIELD_PATHS
    from project.core.feedback import collect_feedback
    from project.core.schemas import TaskRequest, UserConstraints
    from project.orchestrator import CareerOrchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="大学生职业规划助手结题版 CLI")
    parser.add_argument("--session-id", default="default-session")
    parser.add_argument("--goal", required=True, help="职业目标")
    parser.add_argument("--text", default="", help="补充文本")
    parser.add_argument(
        "--docs",
        nargs="*",
        default=[],
        help="TXT/MD/CSV/TSV/PDF/DOCX/XLSX 文档路径",
    )
    parser.add_argument("--images", nargs="*", default=[], help="可选图像路径")
    parser.add_argument("--audio", nargs="*", default=[], help="可选音频路径")
    parser.add_argument("--city", default=None)
    parser.add_argument("--education", default=None)
    parser.add_argument("--time-budget", type=float, default=None)
    parser.add_argument("--financial-budget", type=int, default=None)
    parser.add_argument("--brain-model", default=None)
    parser.add_argument("--answers-json", help="已有八维度答案的 JSON 对象；仅补问必要缺口")
    parser.add_argument("--no-follow-up", action="store_true", help="跳过本地需求引导，保留未解决项")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--debug-trace", action="store_true")
    return parser


def _parse_answers(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("--answers-json 必须是 JSON 对象")
    if any(key not in FIELD_PATHS or not isinstance(item, str) for key, item in value.items()):
        raise ValueError("--answers-json 仅支持八个画像字段，答案必须是字符串")
    return value


def main(
    argv: Sequence[str] | None = None,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    orchestrator_factory: Callable[[], CareerOrchestrator] = CareerOrchestrator,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        answers = _parse_answers(args.answers_json)
    except (ValueError, json.JSONDecodeError) as exc:
        output_fn(f"参数错误: {exc}")
        return 2

    if args.time_budget is not None and not 0 <= args.time_budget <= 168:
        output_fn("参数错误: 每周时间须在 0 到 168 小时之间")
        return 2

    request = TaskRequest(
        session_id=args.session_id,
        user_goal=args.goal,
        text_input=args.text,
        image_paths=args.images,
        document_paths=args.docs,
        audio_paths=args.audio,
        brain_model=args.brain_model,
        stream=args.stream,
        debug_trace=args.debug_trace,
        follow_up_answers=answers,
        skip_follow_up=args.no_follow_up,
        constraints=UserConstraints(
            city=args.city,
            education_level=args.education,
            time_budget_hours_per_week=args.time_budget,
            financial_budget_cny=args.financial_budget,
        ),
    )

    orchestrator = orchestrator_factory()
    prepared = orchestrator.prepare(request, input_fn=input_fn, output_fn=output_fn)
    if prepared.profile.guidance.stop_reason == "model_error":
        try:
            choice = input_fn("本地引导未完成。输入‘继续’可带未解决项生成规划，其他输入退出。\n> ")
        except (EOFError, KeyboardInterrupt):
            return 1
        if choice.strip() != "继续":
            return 1
    unresolved = [key for key, state in prepared.profile.guidance.fields.items()
                  if state.status in {"missing", "unknown", "conflict"}]
    if unresolved:
        output_fn("部分信息仍未明确，规划会保留这些限制。")
    if args.stream:
        final_result = None
        for event in orchestrator.run_stream(request, prepared=prepared):
            event_name = event.get("event")
            data = event.get("data", {})
            if event_name == "token":
                sys.stdout.write(str(data.get("token", "")))
                sys.stdout.flush()
            elif event_name == "final_result":
                final_result = data
        result = final_result or {}
    else:
        response = orchestrator.run(request, prepared=prepared)
        result = response.model_dump()
    if result.get("served_by") != "cloud_brain":
        output_fn("DeepSeek 未完成规划，请检查配置或运行错误；不以模板输出冒充最终规划。")
        return 1
    display_result = dict(result)
    if not args.debug_trace:
        display_result.pop("plan_id", None)
        display_result.pop("output_version", None)
    if not args.debug_trace and "profile" in display_result:
        display_result["profile"] = dict(display_result["profile"])
        display_result["profile"].pop("guidance", None)
    output_fn(json.dumps(display_result, ensure_ascii=False, indent=2))
    feedback = collect_feedback(input_fn=input_fn)
    try:
        adaptation = orchestrator.adapt_feedback(args.session_id, feedback)
    except (OSError, ValueError):
        output_fn("反馈或调整结果未能保存，原规划已保留。")
        return 1
    output_fn(f"反馈已记录：{feedback}")
    if adaptation.status == "adapted":
        output_fn(f"调整后的规划（第 {adaptation.output_version.version} 版）：")
        output_fn(adaptation.output_version.display_text)
    elif adaptation.status == "failed":
        output_fn("详略调整未完成，已保留原规划。")
        return 1
    elif adaptation.reason == "length_constraint":
        output_fn("保留必要信息后无法进一步按此方向调整，继续使用原规划。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
