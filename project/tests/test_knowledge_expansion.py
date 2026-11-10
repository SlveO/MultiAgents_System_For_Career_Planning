from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from project.core.career_knowledge import CareerKnowledgeBase, format_knowledge_hint
from project.core.intake import apply_answers_to_profile, initialize_fields
from project.core.knowledge_graph import build_career_graph, render_mermaid, render_report
from project.core.schemas import CareerPlanResponse, Milestone, UserConstraints, UserProfile


def sample_plan() -> CareerPlanResponse:
    profile = apply_answers_to_profile(UserProfile(), {
        "education": "本科大三", "skills": "SQL", "target_role": "数据分析师",
        "constraints": "联系 demo@example.com，每周时间有限",
    })
    return CareerPlanResponse(
        session_id="private-session", plan_id="plan-1", intent="planning", profile=profile,
        target_roles=["数据分析师"], gap_analysis=["项目经验不足"],
        roadmap_30_90_180=[Milestone(period=period, objective=objective, deliverables=["项目清单"], metrics=["1份"])
                          for period, objective in [("30d", "盘点"), ("90d", "完成项目"), ("180d", "投递复盘")]],
        learning_resources=["SQLBolt"], next_actions=["拆解一份JD"], risk_flags=["时间有限"],
        user_facing_advice="先盘点，再做项目。", knowledge_hit_ids=["career-001"], served_by="cloud_brain",
    )


class TestKnowledgeExpansion(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name) / "kb.json"
        self.entry = {
            "id": "career-001", "role": "数据分析师", "skills": ["SQL", "Python", "统计学", "可视化", "业务理解", "沟通", "实验设计"],
            "resources": ["SQLBolt", "统计课程", "公开数据", "练习题", "第五份资源"],
            "transition_paths": ["运营 -> 数据分析", "数据分析 -> 商业分析"],
            "salary_hint": "历史参考范围", "suitable_majors": ["统计学"],
            "sources": ["https://example.com/career?token=private#fragment"],
        }
        path.write_text(json.dumps([self.entry], ensure_ascii=False), encoding="utf-8")
        self.kb = CareerKnowledgeBase(kb_path=str(path))

    def test_full_knowledge_fields_and_cautious_provenance_reach_hint(self) -> None:
        with patch.object(self.kb, "_init_embedder", side_effect=AssertionError("offline only")):
            hits = self.kb.retrieve("SQL 数据分析师")
        hit = hits[0]
        self.assertEqual(hit["item_id"], "career-001")
        self.assertEqual(hit["provenance_status"], "source_lead_only")
        self.assertEqual(hit["verification_status"], "not_verified")
        hint = format_knowledge_hint(hit)
        for text in ["实验设计", "第五份资源", "运营 -> 数据分析", "统计学", "历史参考范围", "career-001", "尚未逐项核实"]:
            self.assertIn(text, hint)
        # Arbitrary metadata cannot promote a source lead into verified evidence.
        self.assertIn("not_verified", format_knowledge_hint({**hit, "verification_status": "verified"}))

    def test_no_match_and_zero_limit_do_not_invent_relevant_knowledge(self) -> None:
        for query in ["", "astronautxyz"]:
            self.assertEqual(self.kb.retrieve(query), [])
            self.assertEqual(self.kb.search(query), [])
        self.assertEqual(self.kb.retrieve("SQL", top_k=0), [])
        self.assertEqual(self.kb.search("SQL", top_k=-1), [])

    def test_full_entries_keep_list_fields_and_requested_id_order(self) -> None:
        entries = self.kb.get_items_by_ids(["missing", "career-001", "career-001"])
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["skills"], self.entry["skills"])
        self.assertEqual(entries[0]["item_id"], "career-001")
        self.assertIn("第五份资源", self.kb.as_dict_list()[0]["resources"])
        self.assertIn("未逐项核实", self.kb.to_hints("SQL")[0])

    def test_graph_preserves_provenance_and_distinguishes_model_from_knowledge(self) -> None:
        response = sample_plan()
        graph = build_career_graph(response, self.kb.get_items_by_ids(response.knowledge_hit_ids))
        refs = {entry["id"]: entry for entry in graph["provenance"]}
        node_ids = {entry["id"] for entry in graph["nodes"]}
        for entry in graph["nodes"] + graph["edges"]:
            self.assertTrue(entry["provenance_ids"])
            self.assertTrue(set(entry["provenance_ids"]) <= refs.keys())
        for edge in graph["edges"]:
            self.assertIn(edge["from"], node_ids)
            self.assertIn(edge["to"], node_ids)
        statuses = {entry["status"] for entry in refs.values()}
        self.assertTrue({"confirmed_input", "model_suggestion", "source_lead_only"} <= statuses)
        matching_roles = [entry for entry in graph["nodes"] if entry["label"] == "数据分析师"]
        self.assertEqual({entry["kind"] for entry in matching_roles}, {"role_suggestion", "knowledge_role"})
        self.assertEqual(len(matching_roles), 2)
        knowledge_refs = [entry for entry in refs.values() if entry["kind"] == "knowledge_item"]
        self.assertEqual(knowledge_refs[0]["reference"], "career-001")
        self.assertEqual(knowledge_refs[0]["urls"], ["https://example.com/career"])

    def test_unretrieved_entries_are_excluded_and_missing_entries_are_explicit(self) -> None:
        response = sample_plan()
        graph = build_career_graph(response, [{**self.entry, "item_id": "unretrieved-002", "role": "无关岗位"}])
        self.assertNotIn("无关岗位", json.dumps(graph, ensure_ascii=False))
        missing = [entry for entry in graph["provenance"] if entry["status"] == "not_loaded"]
        self.assertEqual(missing[0]["reference"], "career-001")

    def test_mermaid_labels_and_ids_cannot_escape_diagram_syntax(self) -> None:
        graph = {
            "nodes": [{"id": 'x\nclick n0 "javascript:alert(1)"', "label": '中文"]\nclick n0 "https://bad"\n```<script>'}],
            "edges": [],
        }
        mermaid = render_mermaid(graph)
        self.assertEqual(len(mermaid.splitlines()), 2)
        self.assertIn('n0["中文#quot;', mermaid)
        self.assertNotIn("```", mermaid)
        self.assertNotIn("<script>", mermaid)
        self.assertNotIn("javascript:", mermaid)

    def test_report_keeps_plan_and_graph_consistent_and_redacts_private_input(self) -> None:
        response = sample_plan()
        response.next_actions.append("从 /home/alice/private/resume.txt 读取，联系 13812345678")
        response.user_facing_advice += " <script>alert(1)</script> ```"
        before = response.model_dump()
        graph = build_career_graph(response, self.kb.get_items_by_ids(response.knowledge_hit_ids))
        report = render_report(response, graph, display_text="当前展示版本")
        self.assertEqual(response.model_dump(), before)
        for text in ["当前展示版本", "数据分析师", "SQLBolt", "盘点", "完成项目", "投递复盘", "模型规划建议", "career-001"]:
            self.assertIn(text, report)
        all_export = json.dumps(graph, ensure_ascii=False) + report
        for private in ["demo@example.com", "13812345678", "/home/alice", "private-session", "token=private", "<script>"]:
            self.assertNotIn(private, all_export)
        self.assertIn("REDACTED_EMAIL", all_export)
        self.assertIn("REDACTED_PATH", all_export)
        self.assertEqual(report.count("```"), 2)

    def test_report_rejects_graph_from_another_plan(self) -> None:
        response = sample_plan()
        graph = build_career_graph(response)
        response.plan_id = "another-plan"
        with self.assertRaisesRegex(ValueError, "different planning output"):
            render_report(response, graph)

    def test_profile_without_evidence_is_visible_but_not_promoted_to_confirmed(self) -> None:
        response = sample_plan()
        response.profile = UserProfile(skills=["SQL"], constraints=UserConstraints(financial_budget_cny=0, city="上海"))
        graph = build_career_graph(response)
        report = render_report(response, graph)
        self.assertIn("已掌握技能：SQL", report)
        self.assertIn("资金预算：0元", report)
        self.assertIn("城市：上海", report)
        profile_sources = [entry for entry in graph["provenance"] if entry["kind"] == "profile_record"]
        self.assertTrue(profile_sources)
        self.assertTrue(all(entry["status"] == "profile_unconfirmed" for entry in profile_sources))

    def test_report_does_not_activate_generated_markdown_images_or_links(self) -> None:
        response = sample_plan()
        response.user_facing_advice = '![remote](https://example.com/pixel) [click](javascript:alert(1))'
        report = render_report(response, build_career_graph(response))
        self.assertNotIn("![remote]", report)
        self.assertNotIn("[click]", report)

    def test_textual_source_leads_survive_and_blank_or_invalid_sources_do_not_count(self) -> None:
        response = sample_plan()
        entry = {**self.entry, "item_id": "career-001", "sources": ["教育部职业介绍（2025版）第3章", "  ", "https://"]}
        graph = build_career_graph(response, [entry])
        source = next(item for item in graph["provenance"] if item["kind"] == "knowledge_item")
        self.assertEqual(source["status"], "source_lead_only")
        self.assertEqual(source["sources"], ["教育部职业介绍（2025版）第3章"])
        self.assertEqual(source["urls"], [])
        self.assertIn("教育部职业介绍", render_report(response, graph))
        for sources in [["", "   "], ["https://", "javascript:alert(1)"], ["/srv/alice/source.txt", "C:/Users/alice/source.txt"]]:
            with self.subTest(sources=sources):
                graph = build_career_graph(response, [{**entry, "sources": sources}])
                source = next(item for item in graph["provenance"] if item["kind"] == "knowledge_item")
                self.assertEqual(source["status"], "unsourced")
                self.assertEqual(source["sources"], [])

    def test_private_srv_and_forward_slash_windows_paths_are_redacted_without_changing_urls(self) -> None:
        response = sample_plan()
        response.next_actions = ["参考/srv/alice/resume.txt 和 C:/Users/alice/resume.docx", "查看 https://example.com//srv/alice/public 和 https://example.com/C:/public"]
        graph = build_career_graph(response)
        exported = json.dumps(graph, ensure_ascii=False) + render_report(response, graph)
        self.assertNotIn("/srv/alice/resume.txt", exported)
        self.assertNotIn("C:/Users/alice", exported)
        self.assertIn("https://example.com//srv/alice/public", exported)
        self.assertIn("https://example.com/C:/public", exported)

    def test_uuid_machine_identity_is_not_changed_by_phone_redaction(self) -> None:
        response = sample_plan()
        response.plan_id = "abcde13812345678abcdef0123456789"
        self.assertEqual(len(response.plan_id), 32)
        graph = build_career_graph(response)
        self.assertEqual(graph["plan_id"], response.plan_id)
        render_report(response, graph)
        response.plan_id = "/srv/alice/private-plan"
        graph = build_career_graph(response)
        self.assertTrue(graph["plan_id"].startswith("plan_"))
        self.assertNotIn("alice", graph["plan_id"])
        render_report(response, graph)

    def test_synthesized_stored_profile_evidence_does_not_confirm_input(self) -> None:
        response = sample_plan()
        response.profile = UserProfile(skills=["SQL"], major="统计学")
        initialize_fields(response.profile)
        graph = build_career_graph(response)
        input_provenance = [item for item in graph["provenance"] if item["kind"] in {"input_evidence", "profile_record"}]
        self.assertTrue(input_provenance)
        self.assertTrue(all(item["status"] == "profile_unconfirmed" for item in input_provenance))

    def test_machine_node_edge_and_provenance_ids_remain_exact_in_report(self) -> None:
        response = sample_plan()
        graph = build_career_graph(response)
        node_id = "n_abc13812345678abcdef"
        edge_id = "e_abc13812345678abcdef"
        provenance_id = "p_abc13812345678abcdef"
        self.assertEqual(len(node_id), 22)
        graph["nodes"] = [{"id": node_id, "kind": "plan", "label": "联系13812345678", "provenance_ids": [provenance_id]}]
        graph["edges"] = [{"id": edge_id, "from": node_id, "to": node_id, "label": "示例关系", "provenance_ids": [provenance_id]}]
        graph["provenance"] = [{"id": provenance_id, "reference": "示例来源", "notice": "未核实"}]
        report = render_report(response, graph)
        for identifier in (node_id, edge_id, provenance_id):
            self.assertIn(identifier, report)
        self.assertNotIn("联系13812345678", report)


if __name__ == "__main__":
    unittest.main()
