"""画像 schema 与画像 Prompt 的静态校验测试。"""
import json
from pathlib import Path

import jsonschema
import pytest

from src.model.prompts.profile import PROFILE_SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "user_profile.schema.json"


def load_schema():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


class TestSchema:
    def test_schema_loads_and_validates_good_profile(self):
        schema = load_schema()
        good = {
            "major": "计算机科学与技术",
            "grade": "大三",
            "skills": ["Python", "C++"],
            "interests": ["人工智能"],
            "career_goals": ["算法工程师"],
            "output_preference": "too_detailed",
            "missing_fields": [],
        }
        jsonschema.validate(good, schema)

    def test_schema_rejects_extra_fields(self):
        schema = load_schema()
        bad = {"major": "计算机", "missing_fields": [], "name": "张三"}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)

    def test_schema_requires_missing_fields(self):
        schema = load_schema()
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"major": "计算机"}, schema)

    def test_schema_allows_empty_output_preference(self):
        schema = load_schema()
        jsonschema.validate({"output_preference": "", "missing_fields": ["major"]}, schema)


class TestPrompt:
    def test_prompt_contains_json_word(self):
        # DeepSeek JSON 模式要求 prompt 中出现 "json" 字样
        assert "JSON" in PROFILE_SYSTEM_PROMPT

    def test_prompt_forbids_fabrication(self):
        assert "禁止臆造" in PROFILE_SYSTEM_PROMPT
        assert "missing_fields" in PROFILE_SYSTEM_PROMPT
