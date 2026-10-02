from __future__ import annotations

import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from project import assistant_cli
from project.core.local_feedback import LocalFeedbackModel
from project.core.output_adaptation import adapt_output, build_catalog, facts_hash, original_version
from project.core.run_logging import JsonlRunLogger
from project.core.schemas import CareerPlanResponse, Milestone, TaskRequest, UserConstraints, UserProfile
from project.core.session_memory import SessionMemory
from project.orchestrator import CareerOrchestrator
from project.tests.test_completion_flow import FakeDeepSeekClient


class FakeFeedbackModel:
    def __init__(self, decision=None):
        self.decision = decision
        self.calls = []
        self.unloaded = False

    def analyze(self, context):
        self.calls.append(context)
        if isinstance(self.decision, Exception):
            raise self.decision
        if self.decision is not None:
            return self.decision(context) if callable(self.decision) else self.decision
        grouped = {}
        required = set(context['required_ids'])
        for item in context['catalog']:
            if item['item_id'] in required:
                grouped.setdefault(item['section'], []).append(item['item_id'])
        return {'sections': [{'section': section, 'item_ids': ids, 'style': 'bullets'}
                             for section, ids in grouped.items()]}

    def unload(self):
        self.unloaded = True


def sample_response(long=False):
    response = CareerPlanResponse(
        session_id='demo', intent='planning', plan_id='plan-1',
        profile=UserProfile(skills=['SQL'], main_constraints=['实习经验不足'],
                            constraints=UserConstraints(time_budget_hours_per_week=0, financial_budget_cny=100)),
        target_roles=['数据分析师'], gap_analysis=['缺少项目成果'],
        roadmap_30_90_180=[Milestone(period=period, objective=objective, deliverables=['可展示材料'], metrics=['1份成果'])
                          for period, objective in [('30d', '盘点能力'), ('90d', '完成项目'), ('180d', '投递复盘')]],
        learning_resources=['SQLBolt'], next_actions=['今天拆解两份JD'], risk_flags=['时间有限'],
        user_facing_advice='先盘点能力，再开展项目。' * (100 if long else 1), served_by='cloud_brain',
    )
    response.output_version = original_version(response)
    return response


class AdaptableBrain(FakeDeepSeekClient):
    def __init__(self, long=False):
        self.long = long
        self.calls = 0

    def plan(self, prompt, model=None):
        self.calls += 1
        data = json.loads(super().plan(prompt, model))
        data['user_facing_advice'] = (
            '你的目标岗位是数据分析师。建议先完成岗位能力盘点，再制作可展示项目。'
            '目前差距是缺少可量化项目，规划围绕能力盘点、项目制作和投递复盘展开。\n\n'
            '30天阶段以完成盘点为目标。交付物是JD清单，检查指标是10份JD。'
            '可将每份JD中的要求整理到清单中，再用清单梳理后续项目需要展示的能力，'
            '这一阶段的产物仍然是JD清单，不把阅读材料本身当成项目成果。\n\n'
            '90天阶段以完成项目为目标。交付物是作品集，检查指标是2个项目。'
            '把项目材料整理到作品集中，让盘点结果和作品之间形成可回看的联系，'
            '展示时围绕已完成的项目材料展开，保留项目与能力要求之间的对应关系。\n\n'
            '180天阶段以完成投递为目标。交付物是复盘表，检查指标是30次投递。'
            '用复盘表记录投递过程，后续查看记录时可以继续围绕已有作品进行整理。\n\n'
            '下一行动是今天拆解两份JD。学习资源为SQLBolt。风险是目标过多，'
            '因此先围绕数据分析师方向推进上述路线，保留盘点、作品集和复盘表供回顾。'
        ) if self.long else '先盘点能力，再开展项目。'
        return json.dumps(data, ensure_ascii=False)


class TestOutputAdaptation(unittest.TestCase):
    def test_three_choices_preserve_facts_and_only_create_visible_version_when_changed(self):
        for feedback, long in [('过短', False), ('过于详细', True), ('合适', False)]:
            with self.subTest(feedback=feedback):
                response = sample_response(long)
                before = response.model_dump()
                model = FakeFeedbackModel()
                result = adapt_output(response, feedback, 'request-1', model)
                self.assertEqual(response.model_dump(), before)
                self.assertEqual(result.output_version.facts_sha256, facts_hash(response))
                if feedback == '合适':
                    self.assertEqual(result.status, 'unchanged')
                    self.assertEqual(result.output_version.display_text, response.user_facing_advice)
                    self.assertFalse(model.calls)
                    self.assertEqual(result.output_version.version, 1)
                else:
                    self.assertEqual(result.status, 'adapted')
                    self.assertEqual(result.output_version.parent_version, 1)
                    self.assertEqual(result.output_version.version, 2)
                    if feedback == '过短':
                        self.assertGreater(len(result.output_version.display_text), len(response.user_facing_advice))
                    else:
                        self.assertLess(len(result.output_version.display_text), len(response.user_facing_advice))
                    for fact in ['数据分析师', '30天目标：盘点能力', '90天目标：完成项目', '180天目标：投递复盘',
                                 '0小时', '100元', '实习经验不足', '时间有限', '今天拆解两份JD']:
                        self.assertIn(fact, result.output_version.display_text)
                    self.assertTrue(model.unloaded)

    def test_invalid_reference_missing_facts_and_free_text_cannot_change_output(self):
        def default(ctx): return FakeFeedbackModel().analyze(ctx)
        def unknown(ctx):
            value = default(ctx)
            value['sections'][0]['item_ids'].append('invented-job:0')
            return value
        def missing(ctx):
            value = default(ctx)
            value['sections'] = [s for s in value['sections'] if s['section'] != 'risks']
            return value
        def duplicate(ctx):
            value = default(ctx)
            value['sections'][0]['item_ids'].append(value['sections'][0]['item_ids'][0])
            return value
        def prose(ctx):
            value = default(ctx)
            value['replacement_text'] = '改投医药负责人，月薪50000元'
            return value
        def wrong_section(ctx):
            value = default(ctx)
            value['sections'][0]['section'] = 'risks'
            return value
        for bad in [unknown, missing, duplicate, prose, wrong_section, {'sections': []}]:
            with self.subTest(bad=str(bad)):
                response = sample_response(True)
                result = adapt_output(response, '过于详细', 'bad', FakeFeedbackModel(bad))
                self.assertEqual(result.status, 'failed')
                self.assertEqual(result.reason, 'invalid_layout')
                self.assertEqual(result.output_version, response.output_version)
                self.assertNotIn('50000', result.output_version.display_text)

    def test_impossible_compression_and_model_errors_keep_original_version(self):
        response = sample_response()
        result = adapt_output(response, '过于详细', 'short', FakeFeedbackModel())
        self.assertEqual((result.status, result.reason), ('unchanged', 'length_constraint'))
        self.assertEqual(result.output_version.version, 1)
        model = FakeFeedbackModel(RuntimeError('/private/model/path'))
        result = adapt_output(response, '过短', 'failed', model)
        self.assertEqual((result.status, result.reason), ('failed', 'model_unavailable'))
        self.assertNotIn('/private', result.model_dump_json())
        self.assertTrue(model.unloaded)

    def test_advice_without_structured_content_cannot_be_expanded_by_adding_headers(self):
        response = CareerPlanResponse(session_id='s', intent='planning', profile=UserProfile(),
                                      plan_id='bare', user_facing_advice='先明确目标。')
        response.output_version = original_version(response)
        model = FakeFeedbackModel()
        result = adapt_output(response, '过短', 'bare', model)
        self.assertEqual((result.status, result.reason), ('unchanged', 'no_content_to_adjust'))
        self.assertFalse(model.calls)

    def test_changed_source_and_roadmap_order_are_guarded(self):
        response = sample_response()
        response.target_roles = ['其他岗位']
        model = FakeFeedbackModel()
        result = adapt_output(response, '过短', 'changed', model)
        self.assertEqual(result.reason, 'source_changed')
        self.assertFalse(model.calls)
        result = adapt_output(response, '合适', 'changed-suitable', model)
        self.assertEqual(result.reason, 'source_changed')
        response = sample_response()
        def reverse(ctx):
            value = FakeFeedbackModel().analyze(ctx)
            for section in value['sections']:
                if section['section'] == 'roadmap':
                    section['item_ids'].reverse()
            return value
        result = adapt_output(response, '过短', 'ordered', FakeFeedbackModel(reverse))
        text = result.output_version.display_text
        self.assertLess(text.index('30天目标'), text.index('90天目标'))
        self.assertLess(text.index('90天目标'), text.index('180天目标'))


class TestFeedbackPersistence(unittest.TestCase):
    def orchestrator(self, root, *, long=False, model=None):
        return CareerOrchestrator(
            db_path=str(root/'sessions.db'), brain_client=AdaptableBrain(long),
            feedback_model=model or FakeFeedbackModel(), run_logger=JsonlRunLogger(root/'runs.jsonl'),
        )

    def test_versions_feedback_and_logs_are_idempotent_in_sqlite_and_json(self):
        for backend in ['sqlite', 'json']:
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                model = FakeFeedbackModel()
                orch = self.orchestrator(root, model=model)
                if backend == 'json':
                    orch.memory.backend = 'json'
                response = orch.run(TaskRequest(session_id='demo', user_goal='求职'))
                original = response.model_dump()
                result = orch.adapt_feedback('demo', '过短')
                duplicate = orch.adapt_feedback('demo', '过短')
                self.assertEqual(result, duplicate)
                self.assertEqual(len(model.calls), 1)
                self.assertEqual(response.model_dump(), original)
                versions = orch.memory.get_output_versions('demo', response.plan_id)
                self.assertEqual([v['version'] for v in versions], [1, 2])
                self.assertEqual(versions[0]['display_text'], response.user_facing_advice)
                log_lines = (root/'runs.jsonl').read_text().splitlines()
                self.assertEqual(len(log_lines), 1)
                log = json.loads(log_lines[0])
                self.assertEqual(log['feedback_adaptation']['output_version'], versions[1])
                self.assertEqual(log['output_versions'], versions)
                if backend == 'sqlite':
                    with orch.memory._connect() as conn:
                        self.assertEqual(conn.execute('select count(*) from feedback').fetchone()[0], 1)
                        self.assertIsNone(conn.execute('select rating from feedback').fetchone()[0])
                else:
                    self.assertEqual(len(orch.memory._load_fallback()['feedback']), 1)
                # Durable retry after process restart never loads a local model or requests a new plan.
                restarted_model = FakeFeedbackModel(RuntimeError('must not load'))
                restarted = self.orchestrator(root, model=restarted_model)
                restarted.memory.backend = backend
                self.assertEqual(restarted.adapt_feedback('demo', '过短'), result)
                self.assertFalse(restarted_model.calls)
                self.assertEqual(len((root/'runs.jsonl').read_text().splitlines()), 1)

    def test_failed_or_suitable_feedback_does_not_add_a_new_version(self):
        for feedback, model, status in [('合适', FakeFeedbackModel(), 'unchanged'),
                                        ('过短', FakeFeedbackModel(RuntimeError('private')), 'failed')]:
            with self.subTest(feedback=feedback), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                orch = self.orchestrator(root, model=model)
                response = orch.run(TaskRequest(session_id='demo', user_goal='求职'))
                result = orch.adapt_feedback('demo', feedback)
                self.assertEqual(result.status, status)
                self.assertEqual(len(orch.memory.get_output_versions('demo', response.plan_id)), 1)
                if feedback == '合适':
                    self.assertFalse(model.calls)

    def test_token_reuse_and_original_version_overwrite_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            orch = self.orchestrator(root)
            response = orch.run(TaskRequest(session_id='demo', user_goal='求职'))
            orch.adapt_feedback('demo', '合适', request_id='same-id')
            with self.assertRaisesRegex(ValueError, 'different input'):
                orch.adapt_feedback('demo', '过短', request_id='same-id')
            changed = response.output_version.model_dump()
            changed['display_text'] = '替换原文'
            with self.assertRaisesRegex(ValueError, 'different content'):
                orch.memory.save_output_version('demo', changed)

    def test_log_failure_can_be_recovered_after_restart_without_regenerating(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            orch = self.orchestrator(root)
            response = orch.run(TaskRequest(session_id='demo', user_goal='求职'))
            with patch.object(orch.run_logger, 'log_run', side_effect=OSError('disk issue')):
                with self.assertRaises(OSError):
                    orch.adapt_feedback('demo', '过短')
            self.assertEqual(len(orch.memory.get_output_versions('demo', response.plan_id)), 2)
            model = FakeFeedbackModel(RuntimeError('must not generate'))
            restarted = self.orchestrator(root, model=model)
            result = restarted.adapt_feedback('demo', '过短')
            self.assertEqual(result.status, 'adapted')
            self.assertFalse(model.calls)
            saved = json.loads((root/'runs.jsonl').read_text())
            self.assertEqual(saved['status'], 'recovered_feedback')
            self.assertEqual([v['version'] for v in saved['output_versions']], [1, 2])

    def test_failed_json_write_preserves_original_data_and_pending_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            orch = self.orchestrator(root)
            orch.memory.backend = 'json'
            response = orch.run(TaskRequest(session_id='demo', user_goal='求职'))
            original_file = orch.memory.fallback_json.read_bytes()
            with patch('project.core.session_memory.os.replace', side_effect=OSError('write failure')):
                with self.assertRaises(OSError):
                    orch.adapt_feedback('demo', '过短')
            self.assertEqual(orch.memory.fallback_json.read_bytes(), original_file)
            self.assertEqual(len(orch.memory.get_output_versions('demo', response.plan_id)), 1)
            self.assertIn('demo', orch._pending_runs)

    def test_new_plan_same_session_is_not_confused_with_previous_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            orch = self.orchestrator(root)
            first = orch.run(TaskRequest(session_id='demo', user_goal='求职'))
            first_result = orch.adapt_feedback('demo', '合适')
            second = orch.run(TaskRequest(session_id='demo', user_goal='另一个计划'))
            replay = orch.adapt_feedback('demo', '合适', plan_id=first.plan_id)
            self.assertEqual(replay, first_result)
            self.assertEqual(orch._pending_runs['demo'][1].plan_id, second.plan_id)
            self.assertNotEqual(first.plan_id, second.plan_id)

    def test_display_versions_and_logs_redact_personal_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            orch = self.orchestrator(root)
            request = TaskRequest(session_id='demo', user_goal='求职',
                                  follow_up_answers={'constraints': '联系 demo@example.com 或 13812345678'})
            response = orch.run(request)
            result = orch.adapt_feedback('demo', '过短')
            self.assertEqual(result.status, 'adapted')
            serialized = json.dumps(orch.memory.get_output_versions('demo', response.plan_id), ensure_ascii=False)
            serialized += (root/'runs.jsonl').read_text()
            self.assertNotIn('demo@example.com', serialized)
            self.assertNotIn('13812345678', serialized)
            self.assertIn('[REDACTED_EMAIL]', serialized)


class TestFeedbackCli(unittest.TestCase):
    def test_cli_displays_new_version_after_both_normal_and_stream_planning(self):
        for stream, feedback, long in [(False, '1', False), (True, '3', True)]:
            with self.subTest(stream=stream), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                orch = CareerOrchestrator(
                    db_path=str(root/'s.db'), brain_client=AdaptableBrain(long),
                    feedback_model=FakeFeedbackModel(), run_logger=JsonlRunLogger(root/'runs.jsonl'),
                )
                output = []
                args = ['--goal', '求职', '--no-follow-up'] + (['--stream'] if stream else [])
                with patch('sys.stdout', new_callable=io.StringIO):
                    code = assistant_cli.main(args, input_fn=lambda _: feedback, output_fn=output.append,
                                              orchestrator_factory=lambda: orch)
                self.assertEqual(code, 0)
                self.assertTrue(any('第 2 版' in line for line in output))
                saved = json.loads((root/'runs.jsonl').read_text())
                self.assertEqual(output[-1], saved['feedback_adaptation']['output_version']['display_text'])
                self.assertNotIn('facts_sha256', ''.join(output))

    def test_cli_failure_preserves_original_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            orch = CareerOrchestrator(
                db_path=str(root/'s.db'), brain_client=AdaptableBrain(),
                feedback_model=FakeFeedbackModel(RuntimeError('/private/model')),
                run_logger=JsonlRunLogger(root/'runs.jsonl'),
            )
            output = []
            code = assistant_cli.main(['--goal', '求职', '--no-follow-up'], input_fn=lambda _: '1',
                                      output_fn=output.append, orchestrator_factory=lambda: orch)
            self.assertEqual(code, 1)
            self.assertTrue(any('保留原规划' in line for line in output))
            self.assertFalse(any('第 2 版' in line for line in output))
            self.assertNotIn('/private', ''.join(output))


class TestLocalFeedbackAdapter(unittest.TestCase):
    def test_adapter_uses_shared_lazy_loader_and_schema_without_loading_in_tests(self):
        adapter = LocalFeedbackModel('', 'cuda:1')
        with patch.object(adapter, 'generate_json', return_value='validated') as generate:
            result = adapter.analyze({'feedback': '过短', 'catalog': [], 'required_ids': []})
        self.assertEqual(result, 'validated')
        messages, schema = generate.call_args.args
        self.assertEqual(schema.__name__, 'FeedbackLayout')
        self.assertIn('不输出新文字', messages[0]['content'])
        self.assertIsNone(adapter._model)
        with self.assertRaisesRegex(RuntimeError, 'existing local weights'):
            adapter.analyze({})
