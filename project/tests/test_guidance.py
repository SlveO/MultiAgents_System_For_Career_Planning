from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from project.core.guidance import collect_evidence, merge_decision, run_guidance
from project.core.intake import apply_answers_to_profile, initialize_fields
from project.core.local_guidance import LocalGuidanceModel
from project.core.run_logging import JsonlRunLogger
from project.core.schemas import GuidanceDecision, PerceptionResult, TaskRequest, UserConstraints, UserProfile
from project.orchestrator import CareerOrchestrator
from project.tests.test_completion_flow import FakeDeepSeekClient


class FakeGuidanceModel:
    def __init__(self, decisions=None):
        self.decisions = iter(decisions) if decisions is not None else None
        self.contexts = []
        self.unloaded = False

    def analyze(self, context):
        self.contexts.append(context)
        if self.decisions is None:
            return GuidanceDecision()
        value = next(self.decisions)
        if isinstance(value, Exception):
            raise value
        return value(context) if callable(value) else value

    def unload(self):
        self.unloaded = True


def observation(context, field, value, origin="text_input"):
    ref = next(e["evidence_id"] for e in context["evidence"] if e["origin"] == origin)
    return {"observations": [{"field": field, "value": value, "evidence_ids": [ref]}]}


class TestGuidance(unittest.TestCase):
    def guide(self, profile=None, *, model=None, inputs=(), max_rounds=4, req=None):
        model = model or FakeGuidanceModel()
        prompts = []
        iterator = iter(inputs)
        request = req or TaskRequest(session_id="test", user_goal="求职")
        result = run_guidance(
            request, profile or UserProfile(), [], model, max_rounds=max_rounds,
            input_fn=lambda prompt: prompts.append(prompt) or next(iterator), output_fn=lambda _: None,
        )
        return result, prompts, model

    def test_statuses_distinguish_missing_unknown_refused_none_and_zero(self):
        result = apply_answers_to_profile(UserProfile(skills=["旧技能"]), {
            "target_role": "不知道", "major": "不想回答", "skills": "无", "time_budget": "0",
        })
        self.assertEqual(result.skills, [])
        self.assertEqual(result.constraints.time_budget_hours_per_week, 0)
        self.assertEqual({k: result.guidance.fields[k].status for k in
                          ("education", "target_role", "major", "skills", "time_budget")},
                         {"education": "missing", "target_role": "unknown", "major": "refused",
                          "skills": "explicit_none", "time_budget": "known"})
        self.assertEqual(len(result.guidance.answers), 4)

    def test_time_ranges_negative_and_invalid_are_not_guessed(self):
        for text in ["每周10到20小时", "-3", "169", "nan", "2天每天3小时"]:
            with self.subTest(text=text):
                profile = apply_answers_to_profile(UserProfile(), {"time_budget": text})
                self.assertIsNone(profile.constraints.time_budget_hours_per_week)
                self.assertEqual(profile.guidance.fields["time_budget"].status, "unknown")
        profile = apply_answers_to_profile(UserProfile(), {"time_budget": "每周 2.5 小时"})
        self.assertEqual(profile.constraints.time_budget_hours_per_week, 2.5)

    def test_empty_unknown_refused_answers_are_not_reasked(self):
        profile, prompts, model = self.guide(inputs=["", "不知道", "不想回答", "无"], max_rounds=4)
        self.assertEqual([a.field for a in profile.guidance.answers],
                         ["target_role", "skills", "time_budget", "education"])
        self.assertEqual(profile.guidance.stop_reason, "round_limit")
        resumed, _, _ = self.guide(profile, inputs=["统计学"], max_rounds=1)
        self.assertEqual(resumed.guidance.answers[-1].field, "major")
        self.assertTrue(model.unloaded)
        self.assertEqual(len(prompts), 4)

    def test_material_facts_skip_questions_and_references_are_required(self):
        model = FakeGuidanceModel([lambda ctx: observation(ctx, "target_role", "数据分析师")])
        result, prompts, _ = self.guide(
            req=TaskRequest(session_id="test", user_goal="求职", text_input="目标是数据分析师"),
            model=model, inputs=["SQL"], max_rounds=1,
        )
        self.assertEqual(result.target_role, "数据分析师")
        self.assertEqual(result.skills, ["SQL"])
        self.assertIn("技能", prompts[0])
        for bad in [
            {"observations": [{"field": "major", "value": "统计学", "evidence_ids": ["invented"]}]},
            lambda ctx: observation(ctx, "major", "不存在的专业"),
            {"observations": [{"field": "salary", "value": "100", "evidence_ids": ["x"]}]},
        ]:
            with self.subTest(bad=str(bad)):
                result, prompts, _ = self.guide(
                    model=FakeGuidanceModel([bad]),
                    req=TaskRequest(session_id="t", user_goal="求职", text_input="统计学"),
                )
                self.assertEqual(result.guidance.stop_reason, "model_error")
                self.assertEqual(result.major, "")
                self.assertEqual(prompts, [])

    def test_empty_observation_from_real_model_failure_is_rejected_atomically(self):
        def invalid(ctx):
            decision = observation(ctx, "skills", "Python")
            decision["observations"].append({"field": "time_budget", "value": "", "evidence_ids": []})
            return decision
        profile = apply_answers_to_profile(UserProfile(), {"major": "统计学"})
        result, prompts, model = self.guide(
            profile, model=FakeGuidanceModel([invalid]),
            req=TaskRequest(session_id="t", user_goal="求职", text_input="我熟悉Python。"),
        )
        self.assertEqual(result.guidance.stop_reason, "model_error")
        self.assertEqual(result.major, "统计学")
        self.assertEqual(result.skills, [])
        self.assertFalse(prompts)
        self.assertTrue(model.unloaded)

    def test_partial_grounding_failure_keeps_confirmed_skill_and_applies_valid_fact(self):
        def mixed(ctx):
            ref = next(e["evidence_id"] for e in ctx["evidence"] if e["origin"] == "text_input")
            return {"observations": [
                {"field": "skills", "value": "Python、SQL", "evidence_ids": [ref]},
                {"field": "interests", "value": "人工智能", "evidence_ids": ["invented-ref"]},
                {"field": "major", "value": "统计学", "evidence_ids": [ref]},
            ]}

        request = TaskRequest(
            session_id="t", user_goal="求职", text_input="熟悉Python和SQL；统计学专业",
        )
        result, prompts, _ = self.guide(
            UserProfile(skills=["Python", "SQL"]), model=FakeGuidanceModel([mixed]), req=request,
            inputs=["数据分析师"], max_rounds=1,
        )
        self.assertEqual(result.guidance.stop_reason, "round_limit")
        self.assertEqual(result.skills, ["Python", "SQL"])
        self.assertEqual(result.interests, [])
        self.assertEqual(result.major, "统计学")
        self.assertEqual(result.guidance.answers[0].raw_answer, "数据分析师")
        self.assertTrue(prompts)

    def test_partial_grounding_warning_does_not_echo_values_or_errors(self):
        messages = []

        def mixed(ctx):
            ref = next(e["evidence_id"] for e in ctx["evidence"] if e["origin"] == "text_input")
            return {"observations": [
                {"field": "skills", "value": "敏感幻觉字段", "evidence_ids": [ref]},
                {"field": "major", "value": "统计学", "evidence_ids": [ref]},
            ]}

        request = TaskRequest(session_id="t", user_goal="求职", text_input="统计学专业")
        result = run_guidance(
            request, UserProfile(), [], FakeGuidanceModel([mixed]), max_rounds=1,
            input_fn=lambda _: "数据分析师", output_fn=messages.append,
        )
        self.assertEqual(result.guidance.stop_reason, "round_limit")
        self.assertEqual(result.major, "统计学")
        self.assertEqual(result.skills, [])
        self.assertEqual(len(messages), 1)
        self.assertNotIn("敏感幻觉字段", messages[0])
        self.assertNotIn("ValueError", messages[0])

    def test_invalid_question_reference_prevents_all_observation_merges(self):
        def invalid_question(ctx):
            ref = next(e["evidence_id"] for e in ctx["evidence"] if e["origin"] == "text_input")
            return {
                "observations": [{"field": "major", "value": "统计学", "evidence_ids": [ref]}],
                "question": {"field": "major", "reason": "clarify", "prompt": "补充专业",
                             "evidence_ids": ["invented-ref"]},
            }

        request = TaskRequest(session_id="t", user_goal="求职", text_input="统计学专业")
        result, prompts, _ = self.guide(model=FakeGuidanceModel([invalid_question]), req=request)
        self.assertEqual(result.guidance.stop_reason, "model_error")
        self.assertEqual(result.major, "")
        self.assertFalse(prompts)

    def test_direct_merge_remains_atomic_for_mixed_valid_and_invalid_observations(self):
        profile = UserProfile()
        initialize_fields(profile)
        request = TaskRequest(session_id="t", user_goal="求职", text_input="统计学专业")
        collect_evidence(profile, request, [])
        ref = next(e.evidence_id for e in profile.guidance.evidence if e.origin == "text_input")
        decision = GuidanceDecision.model_validate({"observations": [
            {"field": "major", "value": "统计学", "evidence_ids": [ref]},
            {"field": "skills", "value": "不存在", "evidence_ids": [ref]},
        ]})
        with self.assertRaisesRegex(ValueError, "not an evidence excerpt"):
            merge_decision(profile, decision)
        self.assertEqual(profile.major, "")
        self.assertEqual(profile.guidance.fields["major"].status, "missing")

    def test_conflict_requires_confirmation_and_stale_evidence_does_not_reopen_it(self):
        decisions = [lambda ctx: observation(ctx, "target_role", "后端工程师")] * 2
        result, prompts, _ = self.guide(
            UserProfile(target_role="数据分析师"), model=FakeGuidanceModel(decisions),
            req=TaskRequest(session_id="test", user_goal="求职", text_input="后端工程师"),
            inputs=["后端工程师", "SQL"], max_rounds=2,
        )
        self.assertIn("不一致", prompts[0])
        self.assertEqual(result.target_role, "后端工程师")
        self.assertEqual(result.guidance.fields["target_role"].status, "known")
        self.assertEqual(result.guidance.answers[1].field, "skills")

    def test_refusing_conflict_keeps_both_sources_and_no_confirmed_value(self):
        result, prompts, _ = self.guide(
            UserProfile(target_role="数据分析师"),
            model=FakeGuidanceModel([lambda ctx: observation(ctx, "target_role", "后端工程师")] * 2),
            req=TaskRequest(session_id="test", user_goal="求职", text_input="后端工程师"),
            inputs=["不想回答", "SQL"], max_rounds=2,
        )
        self.assertEqual(result.target_role, "")
        self.assertEqual(result.guidance.fields["target_role"].status, "conflict")
        self.assertEqual(len(result.guidance.fields["target_role"].values), 2)
        self.assertGreaterEqual(len(result.guidance.fields["target_role"].evidence_ids), 3)
        self.assertNotIn("不一致", prompts[1])

    def test_model_can_clarify_vague_goal_but_not_repeat_user_answer(self):
        def suggest(ctx):
            refs = ctx["profile"]["guidance"]["fields"]["target_role"]["evidence_ids"]
            return {"question": {"field": "target_role", "reason": "clarify",
                                 "prompt": "你想尝试哪个具体岗位？", "evidence_ids": refs}}
        profile = apply_answers_to_profile(UserProfile(), {"target_role": "科技行业"})
        result, prompts, _ = self.guide(profile, model=FakeGuidanceModel([suggest, suggest]),
                                       inputs=["后端工程师", "SQL"], max_rounds=2)
        self.assertIn("具体岗位", prompts[0])
        self.assertIn("技能", prompts[1])
        self.assertEqual(result.target_role, "后端工程师")

    def test_skip_limit_stop_and_errors_are_explicit(self):
        for skip, rounds, reason in [(True, 4, "skipped"), (False, 0, "round_limit")]:
            result, prompts, model = self.guide(
                req=TaskRequest(session_id="t", user_goal="求职", skip_follow_up=skip), max_rounds=rounds,
            )
            self.assertEqual(result.guidance.stop_reason, reason)
            self.assertFalse(model.contexts)
            self.assertFalse(prompts)
        result, _, model = self.guide(inputs=["结束引导"])
        self.assertEqual(result.guidance.stop_reason, "user_stopped")
        self.assertTrue(model.unloaded)
        result, _, model = self.guide(model=FakeGuidanceModel([RuntimeError("private path")]))
        self.assertEqual(result.guidance.stop_reason, "model_error")
        self.assertTrue(model.unloaded)

    def test_complete_profile_needs_no_questions(self):
        profile = apply_answers_to_profile(UserProfile(), {
            "education": "本科", "major": "统计学", "skills": "SQL", "interests": "分析",
            "target_role": "数据分析师", "time_budget": "8", "preference": "无", "constraints": "无",
        })
        result, prompts, _ = self.guide(profile)
        self.assertEqual(result.guidance.stop_reason, "no_eligible_question")
        self.assertFalse(prompts)

    def test_repeated_none_evidence_and_equivalent_hours_do_not_create_conflicts(self):
        request = TaskRequest(session_id="t", user_goal="求职", text_input="无；每周10小时")
        profile = UserProfile(constraints=UserConstraints(time_budget_hours_per_week=10))
        initialize_fields(profile)
        collect_evidence(profile, request, [])
        ref = next(e.evidence_id for e in profile.guidance.evidence if e.origin == "text_input")
        decision = GuidanceDecision.model_validate({"observations": [
            {"field": "skills", "value": "无", "evidence_ids": [ref]},
            {"field": "time_budget", "value": "每周10小时", "evidence_ids": [ref]},
        ]})
        profile = merge_decision(merge_decision(profile, decision), decision)
        self.assertEqual(profile.guidance.fields["skills"].status, "explicit_none")
        self.assertEqual(profile.guidance.fields["time_budget"].status, "known")

    def test_document_content_beyond_initial_quote_is_available(self):
        result = PerceptionResult(modality="document", summary="parsed", raw_output="前言" * 200 + "统计学")
        profile = UserProfile()
        initialize_fields(profile)
        collect_evidence(profile, TaskRequest(session_id="t", user_goal="求职"), [result])
        ref = next(e.evidence_id for e in profile.guidance.evidence if e.origin == "document:0:parsed_text")
        profile = merge_decision(profile, GuidanceDecision.model_validate({"observations": [
            {"field": "major", "value": "统计学", "evidence_ids": [ref]},
        ]}))
        self.assertEqual(profile.major, "统计学")

    def test_conflict_with_many_sources_still_asks_once(self):
        profile = UserProfile(target_role="数据分析师")
        initialize_fields(profile)
        state = profile.guidance.fields["target_role"]
        state.status = "conflict"
        state.values.append("后端工程师")
        state.evidence_ids.extend(f"history-{i}" for i in range(12))
        result, prompts, _ = self.guide(profile, inputs=["不知道"], max_rounds=1)
        self.assertEqual(len(prompts), 1)
        self.assertEqual(result.guidance.fields["target_role"].status, "conflict")


class TestGuidanceIntegration(unittest.TestCase):
    def test_normal_and_stream_reuse_perception_and_trace_answers_into_plan_and_storage(self):
        for stream in (False, True):
            with self.subTest(stream=stream), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                brain = FakeDeepSeekClient()
                model = FakeGuidanceModel()
                orchestrator = CareerOrchestrator(
                    db_path=str(root / "sessions.db"), brain_client=brain, guidance_model=model,
                    run_logger=JsonlRunLogger(root / "runs.jsonl"),
                )
                orchestrator.settings = orchestrator.settings.model_copy(update={"guidance_max_rounds": 2})
                request = TaskRequest(session_id="integration", user_goal="求职", follow_up_answers={"major": "统计学"})
                inputs = iter(["数据分析师", "SQL"])
                with patch.object(orchestrator, '_collect_perception', wraps=orchestrator._collect_perception) as perceive, \
                     patch.object(orchestrator.knowledge, 'retrieve', wraps=orchestrator.knowledge.retrieve) as retrieve:
                    prepared = orchestrator.prepare(request, input_fn=lambda _: next(inputs), output_fn=lambda _: None)
                    if stream:
                        events = list(orchestrator.run_stream(request, prepared=prepared))
                        payload = events[-1]["data"]
                    else:
                        payload = orchestrator.run(request, prepared=prepared).model_dump()
                    perceive.assert_called_once()
                    self.assertIn("数据分析师", retrieve.call_args.args[0])
                    self.assertIn("SQL", retrieve.call_args.args[0])
                self.assertIn('"raw_answer":"SQL"', brain.prompt)
                self.assertIn('"origin":"interactive"', brain.prompt)
                self.assertEqual(payload["profile"]["skills"], ["SQL"])
                orchestrator.submit_feedback("integration", "合适")
                history = orchestrator.memory.get_session_history("integration")[-1]
                log = json.loads((root / "runs.jsonl").read_text())
                self.assertEqual(history["request"]["guidance"]["answers"][-1]["raw_answer"], "SQL")
                self.assertEqual(log["profile"]["guidance"]["answers"][-1]["raw_answer"], "SQL")
                self.assertEqual(log["profile"]["guidance"]["stop_reason"], "round_limit")
                self.assertTrue(log["profile"]["guidance"]["evidence"])
                # Next request restores all fields and statuses without loading local weights.
                resumed = orchestrator._build_profile(TaskRequest(session_id="integration", user_goal="求职"), [])
                self.assertEqual(resumed.major, "统计学")
                self.assertEqual(resumed.skills, ["SQL"])
                self.assertEqual(resumed.target_role, "数据分析师")

    def test_history_constraints_survive_defaults_and_new_conflicts_are_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = CareerOrchestrator(db_path=str(Path(tmp) / "s.db"), brain_client=FakeDeepSeekClient())
            orchestrator.memory.upsert_profile("t", UserProfile(
                major="统计学", education_stage="本科", skills=["SQL"],
                constraints=UserConstraints(time_budget_hours_per_week=10, city="杭州"),
            ).model_dump())
            profile = orchestrator._build_profile(TaskRequest(session_id="t", user_goal="求职"), [])
            self.assertEqual(profile.constraints.time_budget_hours_per_week, 10)
            self.assertEqual(profile.constraints.city, "杭州")
            self.assertEqual(profile.education_stage, profile.current_stage)
            request = TaskRequest(session_id="t", user_goal="求职", constraints=UserConstraints(time_budget_hours_per_week=5))
            profile = orchestrator._build_profile(request, [])
            self.assertEqual(profile.guidance.fields["time_budget"].status, "conflict")
            self.assertIsNone(profile.constraints.time_budget_hours_per_week)

    def test_prepared_input_cannot_be_reused_after_request_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = CareerOrchestrator(db_path=str(Path(tmp) / "s.db"), brain_client=FakeDeepSeekClient())
            request = TaskRequest(session_id="t", user_goal="求职", skip_follow_up=True)
            prepared = orchestrator.prepare(request)
            request.user_goal = "改变目标"
            with self.assertRaisesRegex(ValueError, "does not match"):
                orchestrator.run(request, prepared=prepared)

    def test_guidance_answer_and_evidence_are_redacted_in_both_stores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            orchestrator = CareerOrchestrator(
                db_path=str(root / "s.db"), brain_client=FakeDeepSeekClient(),
                run_logger=JsonlRunLogger(root / "runs.jsonl"),
            )
            request = TaskRequest(session_id="t", user_goal="求职", skip_follow_up=True,
                                  follow_up_answers={"constraints": "联系 demo@example.com 或 13812345678"})
            prepared = orchestrator.prepare(request)
            orchestrator.run(request, prepared=prepared)
            orchestrator.submit_feedback("t", "合适")
            for value in [json.dumps(orchestrator.memory.get_session_history("t"), ensure_ascii=False),
                          (root / "runs.jsonl").read_text()]:
                self.assertNotIn("demo@example.com", value)
                self.assertNotIn("13812345678", value)
                self.assertIn("[REDACTED_EMAIL]", value)
                self.assertIn("evidence_id", value)


class TestLocalGuidancePolicy(unittest.TestCase):
    def test_adapter_is_lazy_and_refuses_missing_weights_cpu_or_unavailable_cuda(self):
        model_class = MagicMock()
        adapter = LocalGuidanceModel("", "cuda:1", model_class=model_class)
        model_class.from_pretrained.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "existing local weights"):
            adapter._load()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "CPU fallback"):
                LocalGuidanceModel(tmp, "cpu")._load()
            torch = MagicMock()
            torch.cuda.is_available.return_value = False
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                LocalGuidanceModel(tmp, "cuda:1", torch_module=torch, model_class=model_class)._load()
        model_class.from_pretrained.assert_not_called()

    def test_adapter_loads_only_existing_files_directly_on_configured_cuda(self):
        torch, model_class, tokenizer_class = MagicMock(), MagicMock(), MagicMock()
        with tempfile.TemporaryDirectory() as tmp:
            adapter = LocalGuidanceModel(tmp, "cuda:1", torch_module=torch,
                                         model_class=model_class, tokenizer_class=tokenizer_class)
            adapter._load()
        kwargs = model_class.from_pretrained.call_args.kwargs
        self.assertTrue(kwargs["local_files_only"])
        self.assertFalse(kwargs["trust_remote_code"])
        self.assertEqual(kwargs["device_map"], {"": "cuda:1"})
        self.assertTrue(tokenizer_class.from_pretrained.call_args.kwargs["local_files_only"])
        model_class.from_pretrained.return_value.to.assert_not_called()
        adapter.unload()
        self.assertIsNone(adapter._model)

    def test_generate_validates_json_and_rejects_oversized_input_without_generation(self):
        adapter = LocalGuidanceModel("unused", "cuda:1", torch_module=MagicMock())
        adapter._model, adapter._tokenizer = MagicMock(), MagicMock()
        inputs = MagicMock()
        inputs.__getitem__.return_value.shape = [1, 12]
        inputs.to.return_value = inputs
        adapter._tokenizer.apply_chat_template.return_value = inputs
        adapter._tokenizer.decode.return_value = '{"observations":[],"question":null}'
        self.assertIsNone(adapter.analyze({}).question)
        inputs.to.assert_called_once_with("cuda:1")
        self.assertFalse(adapter._model.generate.call_args.kwargs["do_sample"])
        adapter._tokenizer.decode.return_value = 'invalid json'
        with self.assertRaises(ValueError):
            adapter.analyze({})
        adapter._model.generate.reset_mock()
        inputs.__getitem__.return_value.shape = [1, 8193]
        with self.assertRaisesRegex(ValueError, "input budget"):
            adapter.analyze({})
        adapter._model.generate.assert_not_called()
