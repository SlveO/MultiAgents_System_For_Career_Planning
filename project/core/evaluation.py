"""Voluntary ratings tied to one saved planning display version."""
from __future__ import annotations

import uuid
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Iterable

from pydantic import BaseModel, ConfigDict, Field, StrictInt


Rating = Annotated[StrictInt, Field(ge=1, le=5)]


class EvaluationScores(BaseModel):
    model_config = ConfigDict(extra="forbid")
    usefulness: Rating | None = None
    clarity: Rating | None = None
    feasibility: Rating | None = None


class EvaluationSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plan_id: str = Field(min_length=1, max_length=128)
    output_version: StrictInt = Field(ge=1)
    scores: EvaluationScores | None = None
    comment: str = Field(default="", max_length=5000)


def make_evaluation(
    submission: EvaluationSubmission | dict, *, evaluation_id: str | None = None,
    created_at: str | None = None,
) -> dict:
    validated = EvaluationSubmission.model_validate(submission)
    return {
        "id": evaluation_id or uuid.uuid4().hex,
        **validated.model_dump(mode="json"),
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }


def summarize_evaluations(evaluations: Iterable[dict]) -> dict:
    """Report observed ratings only; absent scores contribute no zero/default."""
    samples: dict[str, list[int]] = {key: [] for key in ("usefulness", "clarity", "feasibility")}
    count = 0
    for item in evaluations:
        validated = EvaluationSubmission.model_validate({
            key: item[key] for key in ("plan_id", "output_version", "scores", "comment") if key in item
        })
        count += 1
        if validated.scores:
            for key, value in validated.scores.model_dump().items():
                if value is not None:
                    samples[key].append(value)
    return {
        "evaluations": count,
        "metrics": {
            key: {"n": len(values), "mean": sum(values) / len(values) if values else None}
            for key, values in samples.items()
        },
        "notice": "仅统计实际提交的自愿评分；缺失分数不补齐，不代表独立质量评价。",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="汇总导出的实际用户评价，不补齐评分")
    parser.add_argument("file", type=Path, help="评价数组或含 evaluations 数组的 JSON 文件")
    args = parser.parse_args(argv)
    payload = json.loads(args.file.read_text(encoding="utf-8"))
    records = payload.get("evaluations") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        parser.error("JSON 必须包含评价数组")
    print(json.dumps(summarize_evaluations(records), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
