from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class KnowledgeItem:
    role: str
    skills: List[str]
    resources: List[str]
    transition_paths: List[str]
    salary_hint: str


class CareerKnowledgeBase:
    """Career knowledge base with keyword + vector hybrid retrieval.

    Uses ChromaDB for persistent vector storage and sentence-transformers
    (bge-small-zh-v1.5) for Chinese-optimized embeddings. Falls back to
    keyword-only search when the embedding model is unavailable.
    """

    DEFAULT_EMBEDDING_MODEL = "./models/BAAI/bge-small-zh-v1___5"

    def __init__(
        self,
        kb_path: str | None = None,
        embedding_model_path: str | None = None,
        chroma_persist_dir: str | None = None,
        vector_weight: float = 0.7,
    ) -> None:
        if kb_path is None:
            project_root = Path(__file__).resolve().parents[2]
            self.kb_path = project_root / "dataset" / "career_knowledge_base.json"
        else:
            self.kb_path = Path(kb_path)
        self.embedding_model_path = embedding_model_path or self.DEFAULT_EMBEDDING_MODEL
        if chroma_persist_dir is None:
            project_root = Path(__file__).resolve().parents[2]
            self.chroma_dir = str(project_root / "data" / "chroma_db")
        else:
            self.chroma_dir = chroma_persist_dir
        self.vector_weight = vector_weight

        self.items = self._load_items()
        self._embedder: Any = None
        self._embedder_failed: bool = False
        self._chroma_client: Any = None
        self._collection: Any = None
        self._vector_ready: Optional[bool] = None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _init_embedder(self) -> None:
        if self._embedder is not None or self._embedder_failed:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model_path)
        except Exception:
            self._embedder_failed = True

    def _init_vector_store(self) -> None:
        if self._chroma_client is not None:
            return
        try:
            import chromadb
            self._chroma_client = chromadb.PersistentClient(path=self.chroma_dir)
            self._collection = self._chroma_client.get_or_create_collection(
                name="career_knowledge",
                metadata={"hnsw:space": "cosine"},
            )
            if self._collection.count() == 0 and self.items:
                self._build_vector_index()
        except Exception:
            self._chroma_client = None
            self._collection = None

    @property
    def vector_available(self) -> bool:
        if self._vector_ready is None:
            self._init_embedder()
            self._init_vector_store()
            self._vector_ready = (
                self._embedder is not None
                and self._collection is not None
            )
        return self._vector_ready

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

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

    def _save_items(self) -> None:
        self.kb_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "role": item.role,
                "skills": item.skills,
                "resources": item.resources,
                "transition_paths": item.transition_paths,
                "salary_hint": item.salary_hint,
            }
            for item in self.items
        ]
        self.kb_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

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
        self.kb_path.write_text(
            json.dumps(seed, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Vector index
    # ------------------------------------------------------------------

    def _item_to_text(self, item: KnowledgeItem) -> str:
        return " ".join(
            [item.role] + item.skills + item.transition_paths
        )

    def _build_vector_index(self) -> None:
        if self._embedder is None or self._collection is None:
            return
        texts = [self._item_to_text(item) for item in self.items]
        embeddings = self._embedder.encode(texts, show_progress_bar=False)
        metadatas = [
            {
                "role": item.role,
                "skills": ", ".join(item.skills),
                "resources": ", ".join(item.resources),
                "transition_paths": ", ".join(item.transition_paths),
                "salary_hint": item.salary_hint,
            }
            for item in self.items
        ]
        ids = [f"item_{i}" for i in range(len(self.items))]
        self._collection.add(embeddings=embeddings.tolist(), documents=texts, metadatas=metadatas, ids=ids)

    def rebuild_index(self) -> None:
        if self._collection is None:
            return
        try:
            self._collection.delete(ids=self._collection.get()["ids"])
        except Exception:
            pass
        self._build_vector_index()

    # ------------------------------------------------------------------
    # Tokenization and keyword search (fallback)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Vector retrieval
    # ------------------------------------------------------------------

    def _vector_retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        self._init_embedder()
        self._init_vector_store()
        if self._embedder is None or self._collection is None:
            return []
        q_embedding = self._embedder.encode([query], show_progress_bar=False)
        results = self._collection.query(query_embeddings=q_embedding.tolist(), n_results=min(top_k * 2, len(self.items)))
        out: List[Dict[str, Any]] = []
        if not results["ids"] or not results["ids"][0]:
            return out
        for i, item_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results.get("distances") else 1.0
            out.append({
                "item_id": item_id,
                "role": meta.get("role", ""),
                "skills": meta.get("skills", ""),
                "resources": meta.get("resources", ""),
                "transition_paths": meta.get("transition_paths", ""),
                "salary_hint": meta.get("salary_hint", ""),
                "vector_score": round(1.0 - min(distance, 1.0), 4),
            })
        return out

    # ------------------------------------------------------------------
    # Hybrid retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, str]]:
        if not self.vector_available:
            return self._keyword_retrieve(query, top_k)
        return self._hybrid_retrieve(query, top_k)

    def _keyword_retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, str]]:
        q_tokens = set(self._tokenize(query))
        scored = []
        for item in self.items:
            text = " ".join([item.role] + item.skills + item.transition_paths).lower()
            tokens = set(self._tokenize(text))
            overlap = len(q_tokens.intersection(tokens))
            scored.append((overlap, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k] if scored else []

        references: List[Dict[str, str]] = []
        for score, item in top:
            references.append(
                {
                    "role": item.role,
                    "skills": ", ".join(item.skills[:6]),
                    "resources": ", ".join(item.resources[:4]),
                    "transition_paths": ", ".join(item.transition_paths[:3]),
                    "salary_hint": item.salary_hint,
                    "match_score": str(score),
                }
            )
        return references

    def _hybrid_retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, str]]:
        kw_weight = 1.0 - self.vector_weight

        # Keyword results
        q_tokens = set(self._tokenize(query))
        kw_scored: List[tuple[float, KnowledgeItem]] = []
        max_overlap = 1
        for item in self.items:
            text = " ".join([item.role] + item.skills + item.transition_paths).lower()
            tokens = set(self._tokenize(text))
            overlap = len(q_tokens.intersection(tokens))
            if overlap > max_overlap:
                max_overlap = overlap
            kw_scored.append((float(overlap), item))

        # Normalise keyword scores
        kw_normalised: Dict[str, float] = {}
        for score, item in kw_scored:
            norm = score / max(max_overlap, 1)
            kw_normalised[item.role] = norm * kw_weight

        # Vector results
        vec_results = self._vector_retrieve(query, top_k=top_k * 2)

        # Merge scores
        merged: Dict[str, Dict[str, Any]] = {}
        for role, kw_score in kw_normalised.items():
            if kw_score > 0:
                merged[role] = {"role": role, "score": kw_score, "source": "keyword"}

        for vr in vec_results:
            role = vr["role"]
            vec_score = vr["vector_score"] * self.vector_weight
            if role in merged:
                merged[role]["score"] = merged[role]["score"] + vec_score
                merged[role]["source"] = "hybrid"
                merged[role]["vector_score"] = vr["vector_score"]
            else:
                merged[role] = {
                    "role": role,
                    "score": vec_score,
                    "source": "vector",
                    "vector_score": vr["vector_score"],
                }

        # Sort and pick top_k
        ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)

        # Look up full KnowledgeItem for each hit
        item_by_role: Dict[str, KnowledgeItem] = {item.role: item for item in self.items}
        references: List[Dict[str, str]] = []
        for entry in ranked[:top_k]:
            role = entry["role"]
            item = item_by_role.get(role)
            if item is None:
                continue
            references.append(
                {
                    "role": item.role,
                    "skills": ", ".join(item.skills[:6]),
                    "resources": ", ".join(item.resources[:4]),
                    "transition_paths": ", ".join(item.transition_paths[:3]),
                    "salary_hint": item.salary_hint,
                    "match_score": f"{entry['score']:.2f} ({entry['source']})",
                }
            )
        return references

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

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

    def add_item(self, item: KnowledgeItem) -> None:
        self.items.append(item)
        self._save_items()
        if self._collection is not None and self._embedder is not None:
            text = self._item_to_text(item)
            embedding = self._embedder.encode([text], show_progress_bar=False)
            self._collection.add(
                embeddings=embedding.tolist(),
                documents=[text],
                metadatas=[{
                    "role": item.role,
                    "skills": ", ".join(item.skills),
                    "resources": ", ".join(item.resources),
                    "transition_paths": ", ".join(item.transition_paths),
                    "salary_hint": item.salary_hint,
                }],
                ids=[f"item_{len(self.items) - 1}"],
            )

    def add_items(self, items: Sequence[KnowledgeItem]) -> None:
        for item in items:
            self.items.append(item)
        self._save_items()
        self.rebuild_index()
