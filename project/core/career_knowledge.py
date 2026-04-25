from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class KnowledgeItem:
    role: str
    skills: List[str]
    resources: List[str]
    transition_paths: List[str]
    salary_hint: str


class CareerKnowledgeBase:
    def __init__(self, kb_path: str = "../dataset/career_knowledge_base.json"):
        self.kb_path = Path(kb_path)
        self.items = self._load_items()

    def _load_items(self) -> List[KnowledgeItem]:
        if not self.kb_path.exists():
            self._bootstrap_default_kb()

        raw = json.loads(self.kb_path.read_text(encoding="utf-8"))
        items: List[KnowledgeItem] = []
        for obj in raw:
            items.append(
                KnowledgeItem(
                    role=obj.get("role", ""),
                    skills=obj.get("skills", []),
                    resources=obj.get("resources", []),
                    transition_paths=obj.get("transition_paths", []),
                    salary_hint=obj.get("salary_hint", ""),
                )
            )
        return items

    def _bootstrap_default_kb(self) -> None:
        seed = [
            {
                "role": "算法工程师",
                "skills": ["Python", "机器学习", "深度学习", "特征工程", "模型评估"],
                "resources": ["吴恩达机器学习课程", "Hands-On ML", "Kaggle"],
                "transition_paths": ["数据分析师 -> 算法工程师", "后端开发 -> 算法工程师"],
                "salary_hint": "一线城市应届约15k-30k/月",
            },
            {
                "role": "数据分析师",
                "skills": ["SQL", "Python", "统计学", "可视化", "A/B测试"],
                "resources": ["SQLBolt", "Tableau Public", "Kaggle 数据分析项目"],
                "transition_paths": ["运营 -> 数据分析", "BI工程师 -> 数据分析师"],
                "salary_hint": "一线城市应届约10k-20k/月",
            },
            {
                "role": "产品经理",
                "skills": ["需求分析", "原型设计", "沟通协作", "数据驱动", "项目管理"],
                "resources": ["硅谷产品实战课", "启示录", "PRD写作模板"],
                "transition_paths": ["运营 -> 产品经理", "开发 -> 技术产品经理"],
                "salary_hint": "一线城市应届约12k-25k/月",
            },
            {
                "role": "嵌入式工程师",
                "skills": ["C/C++", "RTOS", "驱动开发", "硬件接口", "调试能力"],
                "resources": ["STM32 官方文档", "Linux Device Driver", "嵌入式项目实战"],
                "transition_paths": ["电子信息专业 -> 嵌入式", "自动化专业 -> 嵌入式"],
                "salary_hint": "一线城市应届约12k-22k/月",
            },
            {
                "role": "测试开发工程师",
                "skills": ["Python", "自动化测试", "CI/CD", "接口测试", "质量体系"],
                "resources": ["pytest 文档", "Selenium 文档", "测试开发实战"],
                "transition_paths": ["手工测试 -> 测试开发", "后端开发 -> 测试开发"],
                "salary_hint": "一线城市应届约10k-20k/月",
            },
        ]
        self.kb_path.parent.mkdir(parents=True, exist_ok=True)
        self.kb_path.write_text(json.dumps(seed, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        parts = re.split(r"[\s,，。；;、:/]+", text.lower())
        return [p for p in parts if p]

    def search(self, query: str, top_k: int = 3) -> List[KnowledgeItem]:
        q_tokens = set(self._tokenize(query))
        scored = []
        for item in self.items:
            text = " ".join([item.role] + item.skills + item.transition_paths).lower()
            tokens = set(self._tokenize(text))
            overlap = len(q_tokens.intersection(tokens))
            if overlap > 0:
                scored.append((overlap, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            return self.items[:top_k]
        return [x[1] for x in scored[:top_k]]

    def to_hints(self, query: str, top_k: int = 3) -> List[str]:
        hints = []
        for item in self.search(query, top_k=top_k):
            hints.append(
                f"{item.role} | 核心技能: {', '.join(item.skills[:4])} | 薪资参考: {item.salary_hint}"
            )
        return hints

    def as_dict_list(self) -> List[Dict[str, str]]:
        out = []
        for item in self.items:
            out.append(
                {
                    "role": item.role,
                    "skills": ", ".join(item.skills),
                    "resources": ", ".join(item.resources),
                    "transition_paths": ", ".join(item.transition_paths),
                    "salary_hint": item.salary_hint,
                }
            )
        return out
