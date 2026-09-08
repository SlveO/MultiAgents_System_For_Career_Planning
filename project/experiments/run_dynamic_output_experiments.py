"""Experiment 4 (member B, manual 6): dynamic output adjustment.

Manual experiment 4 compares 统一输出 (uniform output) against 按反馈调整
(feedback-adjusted output): fixed cases, one variable, raw data preserved.

Design (frozen 2026-09-08, prompts planning-v1/feedback-v1):
- 20 fixed anonymized cases (project/experiments/dynamic_output_cases.json),
  10 with 过短 feedback and 10 with 过于详细. 合适 is excluded because it
  triggers no regeneration and therefore equals the uniform condition.
- Each case runs twice: uniform planning output (group A) and one
  feedback-adjusted regeneration (group B). 20 cases x 2 groups = 40 rows.
- Automated records per manual 6.1: input, output, latency, errors,
  retrieval hits, token usage, plus per-tier detail metrics so the tier
  difference is observable (档位差异可辨) and direction match (接收效率).
- Human 5-point scores (信息完整性/建议相关性/实用性/输出适配度) are filled
  in by member C from scores_template.csv; 输出适配度 doubles as 适配度 and
  反馈评分 for the adjusted group.

Data discipline (manual 6.2): offline fake rows are marked offline-fake and
must not count as experiment data; failed rows are kept with their error.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from project.core.brain_client import DeepSeekBrainClient
from project.core.run_logging import JsonlRunLogger
from project.core.schemas import CareerPlanResponse, TaskRequest
from project.orchestrator import CareerOrchestrator

DEFAULT_CASES = Path(__file__).resolve().parent / "dynamic_output_cases.json"
PROMPT_VERSIONS = {"planning": "planning-v1", "feedback": "feedback-v1"}


def _fake_plan_json(advice: str, gap_count: int, deliverable_count: int, action_count: int) -> str:
    return json.dumps(
        {
            "user_facing_advice": advice,
            "target_roles": ["目标岗位"],
            "gap_analysis": [f"差距说明{i}" for i in range(gap_count)],
            "roadmap_30_90_180": [
                {
                    "period": period,
                    "objective": f"{period} 阶段目标",
                    "deliverables": [f"交付物{i}" for i in range(deliverable_count)],
                    "metrics": ["进度指标"],
                }
                for period in ("30d", "90d", "180d")
            ],
            "learning_resources": ["学习资源"],
            "next_actions": [f"行动{i}" for i in range(action_count)],
            "risk_flags": ["风险提示"],
            "follow_up_questions": [],
            "confidence": 0.8,
        },
        ensure_ascii=False,
    )


class FakeTieredDeepSeekClient:
    """Deterministic offline stand-in with observable tier lengths."""

    model_name = "deepseek-v4-flash"
    last_usage: Dict[str, Any] = {}

    def plan(self, prompt: str, model: str | None = None) -> str:
        if "反馈档位: 过短" in prompt:
            return _fake_plan_json("扩充后的建议：从方向收敛到项目证据再到投递复盘，分三个阶段逐步推进。", 4, 3, 5)
        if "反馈档位: 过于详细" in prompt:
            return _fake_plan_json("精简建议。", 1, 1, 1)
        return _fake_plan_json("统一建议：围绕目标岗位补齐项目证据。", 2, 2, 2)

    def plan_stream(self, prompt: str, model: str | None = None) -> Iterable[str]:
        yield self.plan(prompt, model)


def _detail_metrics(response: CareerPlanResponse) -> Dict[str, Any]:
    deliverables = sum(len(item.deliverables) for item in response.roadmap_30_90_180)
    roadmap_metrics = sum(len(item.metrics) for item in response.roadmap_30_90_180)
    content_chars = (
        len(response.user_facing_advice)
        + sum(len(x) for x in response.gap_analysis)
        + sum(len(x) for item in response.roadmap_30_90_180 for x in item.deliverables)
        + sum(len(x) for item in response.roadmap_30_90_180 for x in item.metrics)
        + sum(len(x) for x in response.next_actions)
        + sum(len(x) for x in response.learning_resources)
    )
    return {
        "advice_chars": len(response.user_facing_advice),
        "gap_count": len(response.gap_analysis),
        "deliverable_count": deliverables,
        "metric_count": roadmap_metrics,
        "action_count": len(response.next_actions),
        "resource_count": len(response.learning_resources),
        "content_chars": content_chars,
    }


def _direction_matches(
    feedback: str, uniform: CareerPlanResponse, adjusted: CareerPlanResponse
) -> bool:
    """接收效率 proxy: adjustment moved content in the direction the user asked."""
    uniform_chars = _detail_metrics(uniform)["content_chars"]
    adjusted_chars = _detail_metrics(adjusted)["content_chars"]
    if feedback == "过短":
        return adjusted_chars > uniform_chars
    return adjusted_chars < uniform_chars


def _usage_metrics(usage: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "tokens_prompt": usage.get("prompt_tokens"),
        "tokens_completion": usage.get("completion_tokens"),
        "tokens_total": usage.get("total_tokens"),
    }


def _load_cases(case_file: Path) -> List[Dict[str, Any]]:
    cases = json.loads(case_file.read_text(encoding="utf-8"))
    short = sum(1 for case in cases if case.get("feedback") == "过短")
    detailed = sum(1 for case in cases if case.get("feedback") == "过于详细")
    if len(cases) < 20 or short + detailed != len(cases):
        raise ValueError(
            f"case file must contain >=20 cases using only 过短/过于详细; got {len(cases)}"
        )
    return cases


def _flatten_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flattened = []
    for row in rows:
        metrics = row["metrics"]
        flattened.append(
            {
                "case_id": row["case_id"],
                "group": row["group"],
                "feedback": row["feedback"],
                "brain_backend": row["brain_backend"],
                "served_by": row["served_by"],
                "retry_count": row["retry_count"],
                "latency_ms": row["latency_ms"],
                "knowledge_hits": row["knowledge_hits"],
                "adjust_error": row["adjust_error"],
                **row["usage"],
                **metrics,
                "direction_match": row["direction_match"],
                "adjustment_ratio": row["adjustment_ratio"],
            }
        )
    return flattened


def _write_results(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    (output_dir / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    flat_rows = _flatten_rows(rows)
    with (output_dir / "results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    # Blind-scoring template for member C (manual 6.2 five-point rubric).
    with (output_dir / "scores_template.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["output_id", "case_id", "group", "信息完整性1-5", "建议相关性1-5", "实用性1-5", "输出适配度1-5", "备注"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "output_id": f"{row['case_id']}-{row['group']}",
                    "case_id": row["case_id"],
                    "group": row["group"],
                }
            )


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for tier in ("过短", "过于详细"):
        uniform_rows = [r for r in rows if r["feedback"] == tier and r["group"] == "uniform"]
        adjusted_rows = [r for r in rows if r["feedback"] == tier and r["group"] == "feedback_adjusted"]
        uniform_chars = [_detail_metrics_chars(r) for r in uniform_rows]
        adjusted_chars = [_detail_metrics_chars(r) for r in adjusted_rows]
        match_rate = sum(1 for r in adjusted_rows if r["direction_match"]) / len(adjusted_rows)
        failures = [r for r in adjusted_rows if r["adjust_error"]]
        summary[tier] = {
            "n": len(uniform_rows),
            "uniform_mean_content_chars": round(sum(uniform_chars) / len(uniform_chars), 1),
            "adjusted_mean_content_chars": round(sum(adjusted_chars) / len(adjusted_chars), 1),
            "direction_match_rate": round(match_rate, 2),
            "failure_count": len(failures),
            "failure_cases": [r["case_id"] for r in failures],
        }
    return summary


def _detail_metrics_chars(row: Dict[str, Any]) -> int:
    return int(row["metrics"]["content_chars"])


def _write_chart_svg(output_dir: Path, summary: Dict[str, Any]) -> None:
    """Tiny dependency-free SVG chart: tier difference must be visible."""
    width, height = 760, 360
    left, top = 90, 50
    plot_w, plot_h = 560, 180
    baseline = top + plot_h
    max_chars = max(
        summary[t]["uniform_mean_content_chars"] for t in ("过短", "过于详细")
    )
    max_chars = max(
        max_chars, max(summary[t]["adjusted_mean_content_chars"] for t in ("过短", "过于详细"))
    )
    if max_chars <= 0:
        max_chars = 1

    def bar_x(index: int) -> float:
        slot = plot_w / 4
        return left + slot * index + slot * 0.15

    def bar_w() -> float:
        return (plot_w / 4) * 0.7

    def bar_y(value: float) -> float:
        return baseline - (value / max_chars) * plot_h

    bars = []
    labels = []
    colors = {
        "过短-uniform": "#5B8FF9",
        "过短-feedback_adjusted": "#F6903D",
        "过于详细-uniform": "#5B8FF9",
        "过于详细-feedback_adjusted": "#F6903D",
    }
    order = [
        ("过短", "uniform", "过短·统一"),
        ("过短", "feedback_adjusted", "过短·调整后"),
        ("过于详细", "uniform", "过于详细·统一"),
        ("过于详细", "feedback_adjusted", "过于详细·调整后"),
    ]
    for index, (tier, group, label) in enumerate(order):
        value = summary[tier]["uniform_mean_content_chars" if group == "uniform" else "adjusted_mean_content_chars"]
        x = bar_x(index)
        y = bar_y(value)
        bars.append(
            f'<rect x="{x:.0f}" y="{y:.0f}" width="{bar_w():.0f}" height="{baseline - y:.0f}" fill="{colors[tier + "-" + group]}" />'
        )
        bars.append(
            f'<text x="{x + bar_w() / 2:.0f}" y="{y - 6:.0f}" font-size="12" text-anchor="middle">{value:.0f}</text>'
        )
        labels.append(
            f'<text x="{x + bar_w() / 2:.0f}" y="{baseline + 18:.0f}" font-size="12" text-anchor="middle">{label}</text>'
        )

    match_text = " ｜ ".join(
        f"{tier}档位方向匹配率 {summary[tier]['direction_match_rate']:.0%}"
        for tier in ("过短", "过于详细")
    )
    failure_text = " ｜ ".join(
        f"{tier}失败 {summary[tier]['failure_count']} 例" for tier in ("过短", "过于详细")
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="sans-serif">
  <text x="{left}" y="30" font-size="16" font-weight="bold">实验4 动态输出：平均内容量（统一 vs 按反馈调整，离线汇总口径）</text>
  <line x1="{left}" y1="{baseline}" x2="{left + plot_w}" y2="{baseline}" stroke="#999" stroke-width="1" />
  <text x="{left - 12}" y="{baseline + 4}" font-size="12" text-anchor="end">0</text>
  <text x="{left - 12}" y="{top + 4}" font-size="12" text-anchor="end">{max_chars:.0f}</text>
  {''.join(bars)}
  {''.join(labels)}
  <text x="{left}" y="{baseline + 55}" font-size="13">{match_text}</text>
  <text x="{left}" y="{baseline + 75}" font-size="13">{failure_text}</text>
</svg>
"""
    (output_dir / "chart.svg").write_text(svg, encoding="utf-8")


def _write_summary(output_dir: Path, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 实验4 动态输出 — 汇总（离线口径示例）",
        "",
        f"样例：20 个固定匿名案例 × 2 组 = {len(rows)} 行；单一变量=按反馈调整；Prompt 版本：planning-v1 / feedback-v1。",
        "",
        "| 档位 | 样例数 | 统一平均内容量(字) | 调整后平均内容量(字) | 方向匹配率 | 失败数 |",
        "|---|---|---|---|---|---|",
    ]
    for tier in ("过短", "过于详细"):
        item = summary[tier]
        lines.append(
            f"| {tier} | {item['n']} | {item['uniform_mean_content_chars']} | "
            f"{item['adjusted_mean_content_chars']} | {item['direction_match_rate']:.0%} | {item['failure_count']} |"
        )
    lines.append("")
    lines.append("人工评分：成员 C 按 5 分制填写 scores_template.csv（评分时隐藏组别）。")
    lines.append("数据纪律：离线 fake 行不计入实验数据；失败行保留并注明原因。")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_dynamic_output_experiments(
    output_dir: str | Path = "data/experiments/dynamic_output",
    *,
    live: bool = False,
    case_file: str | Path = DEFAULT_CASES,
) -> List[Dict[str, Any]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cases = _load_cases(Path(case_file))

    brain_client = DeepSeekBrainClient() if live else FakeTieredDeepSeekClient()
    if live and not getattr(brain_client, "api_key", ""):
        raise RuntimeError("--live requires DEEPSEEK_API_KEY")

    orchestrator = CareerOrchestrator(
        db_path=str(output_path / "experiment_sessions.db"),
        brain_client=brain_client,
        run_logger=JsonlRunLogger(output_path / "experiment_runs.jsonl"),
    )

    rows: List[Dict[str, Any]] = []
    for case in cases:
        request_data: Dict[str, Any] = {
            "session_id": f"exp4-{case['case_id']}",
            "user_goal": case["user_goal"],
            "text_input": case["text_input"],
            "follow_up_answers": case["follow_up_answers"],
            "planner_mode": "deepseek",
            "use_knowledge": True,
        }
        request = TaskRequest(**request_data)
        uniform = orchestrator.run(request)
        uniform_usage = dict(getattr(brain_client, "last_usage", {}) or {})
        adjusted, adjust_error = orchestrator.adjust_plan(request.session_id, case["feedback"])
        adjusted_usage = dict(getattr(brain_client, "last_usage", {}) or {})
        if adjusted is None:
            # Manual 3.2: keep the original result; the error is recorded.
            adjusted = uniform

        uniform_metrics = _detail_metrics(uniform)
        adjusted_metrics = _detail_metrics(adjusted)
        direction = None
        ratio: Optional[float] = None
        if adjusted is not uniform:
            direction = _direction_matches(case["feedback"], uniform, adjusted)
            base = uniform_metrics["content_chars"] or 1
            ratio = round(
                (adjusted_metrics["content_chars"] - uniform_metrics["content_chars"]) / base, 3
            )

        for group, response, usage, err in (
            ("uniform", uniform, uniform_usage, ""),
            ("feedback_adjusted", adjusted, adjusted_usage, adjust_error or ""),
        ):
            metrics = _detail_metrics(response)
            rows.append(
                {
                    "case_id": case["case_id"],
                    "group": group,
                    "feedback": case["feedback"],
                    "brain_backend": "live" if live else "offline-fake",
                    "served_by": response.served_by,
                    "retry_count": response.retry_count,
                    "latency_ms": response.latency_ms,
                    "knowledge_hits": len(response.knowledge_hit_ids),
                    "adjust_error": err,
                    "usage": _usage_metrics(usage),
                    "metrics": metrics,
                    "direction_match": direction if group == "feedback_adjusted" else None,
                    "adjustment_ratio": ratio if group == "feedback_adjusted" else None,
                    "output": {
                        "user_facing_advice": response.user_facing_advice,
                        "target_roles": response.target_roles,
                        "gap_analysis": response.gap_analysis,
                        "roadmap_30_90_180": [item.model_dump() for item in response.roadmap_30_90_180],
                        "next_actions": response.next_actions,
                    },
                }
            )

    _write_results(output_path, rows)
    summary = _aggregate(rows)
    _write_summary(output_path, rows, summary)
    _write_chart_svg(output_path, summary)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run manual experiment 4: dynamic output adjustment.")
    parser.add_argument("--output-dir", default="data/experiments/dynamic_output")
    parser.add_argument("--case-file", default=str(DEFAULT_CASES))
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call the configured DeepSeek API instead of the deterministic fake client.",
    )
    args = parser.parse_args()
    rows = run_dynamic_output_experiments(args.output_dir, live=args.live, case_file=args.case_file)
    print(f"Wrote {len(rows)} rows to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
