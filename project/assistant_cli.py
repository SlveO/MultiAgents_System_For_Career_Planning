from __future__ import annotations

import argparse
import json
import sys
from typing import Callable, Sequence

try:
    from .core.intake import collect_follow_up_answers
    from .core.feedback import collect_feedback
    from .core.schemas import TaskRequest, UserConstraints
    from .orchestrator import CareerOrchestrator
except ImportError:
    from project.core.intake import collect_follow_up_answers
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
    parser.add_argument("--time-budget", type=int, default=None)
    parser.add_argument("--financial-budget", type=int, default=None)
    parser.add_argument("--brain-model", default=None)
    parser.add_argument("--answers-json", help="八项追问答案的 JSON 对象，用于可重复演示")
    parser.add_argument("--no-follow-up", action="store_true", help="实验模式：跳过追问")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--debug-trace", action="store_true")
    return parser


def _parse_answers(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("--answers-json 必须是 JSON 对象")
    return {str(key): str(item) for key, item in value.items()}


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

    if not answers and not args.no_follow_up:
        output_fn("请完成 8 个固定追问，用于生成结构化用户画像。")
        answers = collect_follow_up_answers(input_fn=input_fn)

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
    if args.stream:
        final_result = None
        for event in orchestrator.run_stream(request):
            event_name = event.get("event")
            data = event.get("data", {})
            if event_name == "token":
                sys.stdout.write(str(data.get("token", "")))
                sys.stdout.flush()
            elif event_name == "final_result":
                final_result = data
        if final_result is not None:
            output_fn(json.dumps(final_result, ensure_ascii=False, indent=2))
    else:
        response = orchestrator.run(request)
        output_fn(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))
    feedback = collect_feedback(input_fn=input_fn)
    orchestrator.submit_feedback(args.session_id, feedback)
    output_fn(f"反馈已记录：{feedback}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
