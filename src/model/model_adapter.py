"""统一模型适配器：超时/重试/JSON 校验与结构化失败格式。

对上层（A 的编排器）暴露稳定接口：
- generate_json(system_prompt, user_content, json_schema) -> dict
- extract_profile(raw_input) -> {"profile", "errors", "used_fallback", "elapsed_ms", "usage"}

失败返回格式（供 A 写入 SessionState.errors）：
{"module": "model", "code": "MODEL_XXX", "message": "中文提示", "retryable": bool}

重试策略（手册 3.2 "重试一次；仍失败则使用规则提取"）：
任何失败最多重试 1 次（总尝试 2 次）；网络类错误（超时/429/5xx）等待 1 秒后重试，
JSON 错误追加强化约束后重试。认证/余额类错误不重试，直接抛出。
"""

import json
import logging
import re
import time
from pathlib import Path

import jsonschema

from src.model.deepseek_client import (
    DeepSeekClient,
    ModelError,
    ModelJSONError,
    ModelRateLimitError,
    ModelServerError,
    ModelTimeoutError,
)
from src.model.prompts.profile import PROFILE_SYSTEM_PROMPT

logger = logging.getLogger("model")

MAX_ATTEMPTS = 2  # 首次 + 1 次重试
RETRY_WAIT_SECONDS = 1
REINFORCE_MESSAGE = "请严格只输出符合要求的合法 JSON，不要输出任何解释文字或 Markdown 代码围栏。"

_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "user_profile.schema.json"
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")


def load_user_profile_schema():
    """加载画像 JSON Schema（文件缺失时给出清晰错误）。"""
    if not _SCHEMA_PATH.exists():
        raise FileNotFoundError(f"画像 schema 不存在：{_SCHEMA_PATH}")
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def parse_json(content):
    """解析模型输出为 dict；支持 ```json 围栏；失败抛 ModelJSONError。"""
    text = (content or "").strip()
    candidates = [text]
    match = _JSON_FENCE_RE.search(text)
    if match:
        candidates.append(match.group(1).strip())
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except ValueError:
            continue
    raise ModelJSONError("模型输出不是合法 JSON")


def _validate_or_raise(data, schema):
    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as exc:
        raise ModelJSONError(f"模型输出未通过 schema 校验：{exc.message}") from exc


class ModelAdapter:
    """统一模型适配器：负责重试、JSON 解析校验、画像兜底。"""

    def __init__(self, client=None, profile_schema=None):
        self.client = client or DeepSeekClient()
        self.profile_schema = profile_schema or load_user_profile_schema()
        self.last_usage = None

    def generate_json(self, system_prompt, user_content, json_schema):
        """通用 JSON 生成：最多尝试 2 次（重试 1 次），最终失败抛类型化异常。

        返回 dict（已通过 json_schema 校验）；成功调用的 usage 记录在 self.last_usage。
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        last_error = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                content, usage = self.client.chat(messages, json_mode=True)
                data = parse_json(content)
                _validate_or_raise(data, json_schema)
                self.last_usage = usage
                logger.info(
                    "generate_json 成功 attempt=%d total_tokens=%s",
                    attempt, usage.get("total_tokens"),
                )
                return data
            except (ModelTimeoutError, ModelRateLimitError, ModelServerError) as exc:
                last_error = exc
                logger.warning("generate_json 网络类失败 attempt=%d code=%s", attempt, exc.code)
            except ModelJSONError as exc:
                last_error = exc
                logger.warning("generate_json JSON 失败 attempt=%d", attempt)
                messages.append({"role": "user", "content": REINFORCE_MESSAGE})
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_WAIT_SECONDS)
        raise last_error

    def extract_profile(self, raw_input):
        """从原始输入提取用户画像。永不抛异常；最差返回规则兜底画像。

        返回 {"profile": {...}, "errors": [...], "used_fallback": bool,
              "elapsed_ms": int, "usage": dict|None}
        """
        started = time.perf_counter()
        self.last_usage = None
        try:
            data = self.generate_json(PROFILE_SYSTEM_PROMPT, raw_input, self.profile_schema)
        except ModelError as exc:
            logger.warning("extract_profile 降级到规则提取 code=%s", exc.code)
            return {
                "profile": _rule_based_profile(raw_input),
                "errors": [exc.to_error_dict()],
                "used_fallback": True,
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
                "usage": None,
            }
        return {
            "profile": _coerce_profile(data),
            "errors": [],
            "used_fallback": False,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
            "usage": self.last_usage,
        }


PROFILE_FIELDS = ["major", "grade", "skills", "interests", "career_goals", "output_preference"]
_LIST_FIELDS = {"skills", "interests", "career_goals"}
_MAJOR_RE = re.compile(
    r"(人工智能|计算机|软件工程|数据科学|电子信息|通信工程|自动化|电气工程|机械|材料|"
    r"化学|生物|医学|药学|数学|统计|物理|法学|经济|金融|会计|管理|市场营销|中文|"
    r"外语|英语|日语|教育|心理|新闻|设计|美术|音乐|建筑|土木)"
)
_GRADE_RE = re.compile(r"(大一|大二|大三|大四|研一|研二|研三|博一|博二|博三|已毕业|毕业[0-9一二三四五]年)")
_SKILL_RE = re.compile(r"(?:会|熟练|掌握|精通|熟悉|了解)(?:使用|运用)?[:：]?([^，。；、！？\n]{1,20})")
_GOAL_RE = re.compile(r"(?:想做|想当|想成为|目标是|目标岗位是|希望从事|希望成为|打算做|打算从事)[:：]?([^，。；、！？\n]{1,20})")


def _coerce_profile(data):
    """补齐缺省字段；空字段自动加入 missing_fields（双重保险，绝不虚构）。"""
    profile = {}
    for key in PROFILE_FIELDS:
        value = data.get(key)
        if key in _LIST_FIELDS:
            profile[key] = value if isinstance(value, list) else []
        else:
            profile[key] = value if isinstance(value, str) else ""
    missing = set(data.get("missing_fields") or [])
    for key in PROFILE_FIELDS:
        is_empty = not profile[key]
        if is_empty:
            missing.add(key)
    profile["missing_fields"] = sorted(missing)
    return profile


def _rule_based_profile(raw_input):
    """规则提取兜底（手册 3.2）：提取不到的内容一律置空并写入 missing_fields，绝不虚构。"""
    text = raw_input or ""
    major_match = _MAJOR_RE.search(text)
    grade_match = _GRADE_RE.search(text)
    partial = {
        "major": major_match.group(1) if major_match else "",
        "grade": grade_match.group(1) if grade_match else "",
        "skills": list(dict.fromkeys(_SKILL_RE.findall(text)))[:10],
        "interests": [],
        "career_goals": _GOAL_RE.findall(text)[:5],
        "output_preference": "",
        "missing_fields": [],
    }
    return _coerce_profile(partial)
