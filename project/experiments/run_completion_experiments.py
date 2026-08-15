from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from project.core.brain_client import DeepSeekBrainClient
from project.core.run_logging import JsonlRunLogger
from project.core.schemas import CareerPlanResponse, TaskRequest
from project.orchestrator import CareerOrchestrator


class FakeDeepSeekClient:
    """Deterministic offline stand-in used only by the experiment runner."""

    model_name = "deepseek-v4-flash"

    def plan(self, prompt: str, model: str | None = None) -> str:
        return json.dumps(
            {
                "user_facing_advice": "围绕目标岗位补齐项目证据，并按阶段复盘投递结果。",
                "target_roles": ["数据分析师"],
                "gap_analysis": ["缺少可量化项目", "岗位案例表达不足"],
                "roadmap_30_90_180": [
                    {
                        "period": "30d",
                        "objective": "完成岗位与能力盘点",
                        "deliverables": ["岗位 JD 清单", "能力差距表"],
                        "metrics": ["分析 10 份 JD"],
                    },
                    {
                        "period": "90d",
                        "objective": "完成作品集项目",
                        "deliverables": ["两个数据分析案例"],
                        "metrics": ["完成两次项目复盘"],
                    },
                    {
                        "period": "180d",
                        "objective": "完成投递与面试迭代",
                        "deliverables": ["投递看板", "面试复盘表"],
                        "metrics": ["完成 30 次有效投递"],
                    },
                ],
                "learning_resources": ["SQLBolt", "Kaggle"],
                "next_actions": ["今天拆解两份目标岗位 JD"],
                "risk_flags": ["学习输入多但项目输出少"],
                "follow_up_questions": [],
                "confidence": 0.82,
            },
            ensure_ascii=False,
        )

    def plan_stream(self, prompt: str, model: str | None = None) -> Iterable[str]:
        yield self.plan(prompt, model)


FOLLOW_UP_ANSWERS = {
    "education": "本科大三",
    "major": "统计学",
    "skills": "Python、SQL",
    "interests": "数据分析、互联网",
    "target_role": "数据分析师",
    "time_budget": "每周 10 小时",
    "preference": "上海互联网行业",
    "constraints": "项目经验不足",
}


def _score_response(response: CareerPlanResponse) -> Dict[str, Any]:
    complete_sections = [
        response.target_roles,
        response.gap_analysis,
        response.roadmap_30_90_180,
        response.next_actions,
        response.risk_flags,
        response.user_facing_advice,
    ]
    profile_values = [
        response.profile.education_stage,
        response.profile.major,
        response.profile.skills,
        response.profile.interests,
        response.profile.target_role,
        response.profile.preference,
        response.profile.main_constraints,
    ]
    deliverable_count = sum(len(item.deliverables) for item in response.roadmap_30_90_180)
    action_count = deliverable_count + len(response.next_actions)
    return {
        "completeness": round(sum(bool(value) for value in complete_sections) / len(complete_sections), 2),
        "personalization": round(sum(bool(value) for value in profile_values) / len(profile_values), 2),
        "actionability": round(min(action_count / 7, 1.0), 2),
        "latency_ms": response.latency_ms,
        "user_feedback": "合适",
    }


def _flatten_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flattened = []
    for row in rows:
        flattened.append(
            {
                "experiment": row["experiment"],
                "variant": row["variant"],
                **row["configuration"],
                **row["metrics"],
            }
        )
    return flattened


def _write_results(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    (output_dir / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    flat_rows = _flatten_rows(rows)
    with (output_dir / "results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)


def run_completion_experiments(
    output_dir: str | Path = "data/experiments",
    *,
    live: bool = False,
) -> List[Dict[str, Any]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    document = output_path / "experiment_profile.txt"
    document.write_text("做过校园消费数据分析，使用 Python、SQL 和可视化。", encoding="utf-8")

    brain_client = DeepSeekBrainClient() if live else FakeDeepSeekClient()
    if live and not getattr(brain_client, "api_key", ""):
        raise RuntimeError("--live requires DEEPSEEK_API_KEY")

    orchestrator = CareerOrchestrator(
        db_path=str(output_path / "experiment_sessions.db"),
        brain_client=brain_client,
        run_logger=JsonlRunLogger(output_path / "experiment_runs.jsonl"),
    )

    cases = [
        ("template_vs_deepseek", "template", {"planner_mode": "template"}),
        ("template_vs_deepseek", "deepseek", {"planner_mode": "deepseek"}),
        ("without_vs_with_knowledge", "without_knowledge", {"use_knowledge": False}),
        ("without_vs_with_knowledge", "with_knowledge", {"use_knowledge": True}),
        ("text_vs_text_document", "text_only", {"document_paths": []}),
        ("text_vs_text_document", "text_and_document", {"document_paths": [str(document)]}),
        ("without_vs_with_follow_up", "without_follow_up", {"follow_up_answers": {}, "skip_follow_up": True}),
        ("without_vs_with_follow_up", "with_follow_up", {"follow_up_answers": FOLLOW_UP_ANSWERS}),
    ]

    rows: List[Dict[str, Any]] = []
    for index, (experiment, variant, overrides) in enumerate(cases, start=1):
        request_data: Dict[str, Any] = {
            "session_id": f"experiment-{index:02d}",
            "user_goal": "获得数据分析实习",
            "text_input": "希望制定 30、90、180 天职业计划。",
            "follow_up_answers": FOLLOW_UP_ANSWERS,
            "planner_mode": "deepseek",
            "use_knowledge": True,
        }
        request_data.update(overrides)
        response = orchestrator.run(TaskRequest(**request_data))
        rows.append(
            {
                "experiment": experiment,
                "variant": variant,
                "configuration": {
                    "planner_mode": request_data["planner_mode"],
                    "use_knowledge": request_data["use_knowledge"],
                    "has_document": bool(request_data.get("document_paths")),
                    "has_follow_up": bool(request_data.get("follow_up_answers")),
                    "brain_backend": "live" if live else "offline-fake",
                },
                "metrics": _score_response(response),
            }
        )

    _write_results(output_path, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run four completion-MVP comparison experiments.")
    parser.add_argument("--output-dir", default="data/experiments")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call the configured DeepSeek API instead of the deterministic fake client.",
    )
    args = parser.parse_args()
    rows = run_completion_experiments(args.output_dir, live=args.live)
    print(f"Wrote {len(rows)} rows to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
