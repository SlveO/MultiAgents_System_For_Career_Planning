from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Callable, Sequence

try:
    from .core.intake import FIELD_PATHS
    from .core.feedback import collect_feedback
    from .core.schemas import CareerPlanResponse, TaskRequest, UserConstraints
    from .core.report_export import export_report
    from .core.privacy import session_key
    from .core.output_preferences import OutputPreferences
    from .orchestrator import CareerOrchestrator
except ImportError:
    from project.core.intake import FIELD_PATHS
    from project.core.feedback import collect_feedback
    from project.core.schemas import CareerPlanResponse, TaskRequest, UserConstraints
    from project.core.report_export import export_report
    from project.core.privacy import session_key
    from project.core.output_preferences import OutputPreferences
    from project.orchestrator import CareerOrchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="大学生职业规划助手结题版 CLI")
    parser.add_argument("--session-id", default="default-session")
    parser.add_argument("--goal", default="", help="职业目标（生成新规划时必填）")
    parser.add_argument("--text", default="", help="补充文本")
    parser.add_argument(
        "--docs",
        nargs="*",
        default=[],
        help="TXT/MD/CSV/TSV/PDF/DOCX/XLSX 文档路径",
    )
    parser.add_argument("--images", nargs="*", default=[], help="可选图像路径")
    parser.add_argument("--audio", nargs="*", default=[], help="可选音频路径")
    parser.add_argument("--video", nargs="*", default=[], help="可选视频路径，抽帧后由本地视觉模型解析")
    parser.add_argument("--scan-pdf", action="store_true", help="对扫描PDF空文本页启用本地视觉解析")
    parser.add_argument("--feedback-rounds", type=int, default=1, help="最多反馈轮数，1至16；合适或结束时停止")
    parser.add_argument("--resume-feedback", action="store_true", help="继续已保存规划的反馈，不请求新规划")
    parser.add_argument("--plan-id", help="继续反馈时指定本会话的规划，默认最近一份")
    parser.add_argument("--export-dir", help="导出脱敏文字和图谱报告到新目录，已有目录不覆盖")
    parser.add_argument("--detail", choices=["brief", "standard", "detailed"], help="初始规划详略：简要、标准或详细")
    parser.add_argument("--knowledge-level", choices=["unspecified", "beginner", "familiar"], help="自选讲解背景，不作为能力测评")
    parser.add_argument("--presentation", choices=["bullets", "paragraph"], help="初始呈现：分点或分段")
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


def _display_version(version, output_fn: Callable[[str], None]) -> None:
    """Use the persisted reading projection; legacy versions remain unchanged."""
    data = version if isinstance(version, dict) else version.model_dump()
    view = data.get("reading_view") or {}
    if view.get("mode") == "brief":
        excerpt = view.get("excerpt", "")
        output_fn(f"简要正文 {len(excerpt)} 字符；完整正文 {len(data['display_text'])} 字符。")
        output_fn(excerpt)
        if view.get("notice"):
            output_fn(view["notice"])
    else:
        output_fn(data["display_text"])


def main(
    argv: Sequence[str] | None = None,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    orchestrator_factory: Callable[[], CareerOrchestrator] = CareerOrchestrator,
) -> int:
    args = build_parser().parse_args(argv)
    args.session_id = session_key(args.session_id)
    try:
        answers = _parse_answers(args.answers_json)
    except (ValueError, json.JSONDecodeError) as exc:
        output_fn(f"参数错误: {exc}")
        return 2

    if args.time_budget is not None and not 0 <= args.time_budget <= 168:
        output_fn("参数错误: 每周时间须在 0 到 168 小时之间")
        return 2
    if not 1 <= args.feedback_rounds <= 16:
        output_fn("参数错误: 反馈轮数须在 1 到 16 之间")
        return 2
    if not args.resume_feedback and not args.goal.strip():
        output_fn("参数错误: 生成新规划须提供 --goal")
        return 2
    if args.plan_id and not args.resume_feedback:
        output_fn("参数错误: --plan-id 仅用于 --resume-feedback")
        return 2
    if args.export_dir and Path(args.export_dir).exists():
        output_fn("导出目录已存在，请指定新目录。")
        return 2
    preferences = None
    if any(value is not None for value in [args.detail, args.knowledge_level, args.presentation]):
        if args.resume_feedback:
            output_fn("参数错误: 初始表达偏好用于新规划；继续反馈请使用详略评价。")
            return 2
        preferences = OutputPreferences(
            detail=args.detail or "standard", knowledge_level=args.knowledge_level or "unspecified",
            presentation=args.presentation or "bullets",
        )

    if args.resume_feedback:
        orchestrator = orchestrator_factory()
        try:
            response = orchestrator.get_saved_plan(args.session_id, args.plan_id)
            if response.served_by != "cloud_brain":
                output_fn("该记录没有成功的 DeepSeek 规划，不能继续反馈。")
                return 1
            versions = orchestrator.memory.get_output_versions(args.session_id, response.plan_id)
            if not versions:
                raise ValueError("No saved display version")
            _display_version(versions[-1], output_fn)
        except (OSError, ValueError):
            output_fn("无法读取本会话的已保存规划。")
            return 1
        return _finish_plan(args, response, orchestrator, input_fn, output_fn)

    request = TaskRequest(
        session_id=args.session_id,
        user_goal=args.goal,
        text_input=args.text,
        image_paths=args.images,
        document_paths=args.docs,
        audio_paths=args.audio,
        video_paths=args.video,
        brain_model=args.brain_model,
        stream=args.stream,
        debug_trace=args.debug_trace,
        follow_up_answers=answers,
        skip_follow_up=args.no_follow_up,
        metadata={"scan_pdf": "true"} if args.scan_pdf else {},
        output_preferences=preferences,
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
                if not (preferences and preferences.detail == "brief" and not args.debug_trace):
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
    if preferences is not None and not args.debug_trace:
        _display_version(result["output_version"], output_fn)
    else:
        output_fn(json.dumps(display_result, ensure_ascii=False, indent=2))
    return _finish_plan(args, CareerPlanResponse.model_validate(result), orchestrator, input_fn, output_fn)


def _finish_plan(args, response, orchestrator, input_fn, output_fn) -> int:
    for _ in range(args.feedback_rounds):
        try:
            feedback = collect_feedback(input_fn=input_fn, allow_finish=args.feedback_rounds > 1 or args.resume_feedback)
        except (EOFError, KeyboardInterrupt, StopIteration):
            break
        if feedback is None:
            break
        try:
            # Explicit tokens allow repeated choices in later rounds; callers
            # retrying an API-level operation can still supply the same token.
            adaptation = orchestrator.adapt_feedback(
                args.session_id, feedback, plan_id=response.plan_id, request_id=uuid.uuid4().hex,
            )
        except (OSError, ValueError):
            output_fn("反馈或调整结果未能保存，原规划已保留。")
            return 1
        output_fn(f"反馈已记录：{feedback}")
        if adaptation.status == "adapted":
            if adaptation.reason == "brief_view":
                output_fn(f"已切换简略阅读（第 {adaptation.output_version.version} 版）：")
            elif adaptation.reason == "full_view":
                output_fn(f"已恢复完整阅读（第 {adaptation.output_version.version} 版）：")
            else:
                output_fn(f"调整后的规划（第 {adaptation.output_version.version} 版）：")
            _display_version(adaptation.output_version, output_fn)
        elif adaptation.status == "failed":
            output_fn("详略调整未完成，已保留原规划。")
            return 1
        elif adaptation.reason == "length_constraint":
            output_fn("保留必要信息后无法进一步按此方向调整，继续使用原规划。")
        elif adaptation.reason == "already_brief":
            output_fn("当前已是简略阅读。")
        if feedback == "合适":
            break
    if args.export_dir:
        try:
            versions = orchestrator.memory.get_output_versions(args.session_id, response.plan_id)
            export_report(
                response, [],
                args.export_dir, display_text=versions[-1]["display_text"] if versions else None,
            )
        except (OSError, ValueError, UnicodeError):
            output_fn("报告导出未完成，已保存的规划和反馈仍可恢复。")
            return 1
        output_fn("规划文字和图谱报告已导出。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
