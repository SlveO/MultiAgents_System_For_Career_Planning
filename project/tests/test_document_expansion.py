from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from project.agents.perception.document_agent import DocumentPerceptionAgent
from project.agents.perception.video_agent import VideoPerceptionAgent
from project.core.schemas import PerceptionResult


class TestDocumentExpansion(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def _blank_pdf(self, pages=1):
        from pypdf import PdfWriter

        path = self.root / 'scanned.pdf'
        writer = PdfWriter()
        for _ in range(pages):
            writer.add_blank_page(width=612, height=792)
        with path.open('wb') as handle:
            writer.write(handle)
        return path

    def test_docx_tables_preserve_paragraph_and_cell_order(self):
        import docx

        document = docx.Document()
        document.add_paragraph('材料开始')
        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).text = '技能'
        table.cell(0, 1).text = 'Python和SQL'
        table.cell(1, 0).text = '每周时间'
        table.cell(1, 1).text = '8小时'
        document.add_paragraph('材料结束')
        path = self.root / 'profile.docx'
        document.save(path)

        result = DocumentPerceptionAgent().perceive(str(path))

        self.assertGreater(result.confidence, 0)
        self.assertLess(result.raw_output.index('材料开始'), result.raw_output.index('技能 | Python和SQL'))
        self.assertLess(result.raw_output.index('每周时间 | 8小时'), result.raw_output.index('材料结束'))
        self.assertTrue(any('#table=1&row=2' in item.source for item in result.evidence))
        self.assertFalse(result.missing_info)

    def test_parser_failure_has_no_facts_or_private_exception_details(self):
        path = self.root / 'broken.docx'
        path.write_bytes(b'invalid office document')
        result = DocumentPerceptionAgent().perceive(str(path))
        self.assertEqual(result.confidence, 0)
        self.assertEqual(result.facts, [])
        self.assertEqual(result.evidence, [])
        self.assertNotIn(str(self.root), ' '.join(result.missing_info))
        self.assertIn('Failed to parse', result.summary)

    def test_nested_docx_table_content_is_not_omitted(self):
        import docx

        document = docx.Document()
        outer = document.add_table(rows=1, cols=1)
        outer.cell(0, 0).text = '外层材料'
        nested = outer.cell(0, 0).add_table(rows=1, cols=1)
        nested.cell(0, 0).text = '内层技能Python'
        path = self.root / 'nested.docx'
        document.save(path)
        result = DocumentPerceptionAgent().perceive(str(path))
        self.assertIn('内层技能Python', result.raw_output)
        self.assertLess(result.raw_output.index('外层材料'), result.raw_output.index('内层技能Python'))

    def test_text_budget_is_reported_and_retains_only_supported_facts(self):
        path = self.root / 'long.txt'
        path.write_text('Python。' * 40 + '隐藏尾部事实', encoding='utf-8')
        result = DocumentPerceptionAgent(max_text_chars=30).perceive(str(path))
        self.assertEqual(len(result.raw_output), 30)
        self.assertNotIn('隐藏尾部事实', result.raw_output)
        self.assertTrue(any('truncated' in item for item in result.missing_info))
        self.assertTrue(all(fact in result.raw_output for fact in result.facts))

    def test_xlsx_limits_are_visible_and_blank_columns_do_not_shift(self):
        import openpyxl

        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet['A1'] = '岗位'
        sheet['C1'] = '后端开发'
        sheet['M1'] = '超出列限制'
        sheet['A21'] = '超出行限制'
        for number in range(3):
            workbook.create_sheet(f'附表{number}')
        path = self.root / 'profile.xlsx'
        workbook.save(path)
        workbook.close()

        result = DocumentPerceptionAgent().perceive(str(path))

        self.assertIn('岗位 |  | 后端开发', result.raw_output)
        self.assertNotIn('超出', result.raw_output)
        warnings = ' '.join(result.missing_info)
        self.assertIn('rows truncated', warnings)
        self.assertIn('columns truncated', warnings)
        self.assertIn('sheets', warnings)

    def test_scan_vision_is_never_called_without_explicit_injection(self):
        path = self._blank_pdf()
        with patch.object(DocumentPerceptionAgent, '_read_scanned_page') as vision:
            result = DocumentPerceptionAgent().perceive(str(path))
        vision.assert_not_called()
        self.assertEqual(result.confidence, 0)
        self.assertEqual(result.facts, [])
        self.assertIn('--scan-pdf', ' '.join(result.missing_info))

    def test_missing_optional_renderer_is_clear_and_not_a_fact(self):
        path = self._blank_pdf()
        with patch.dict(sys.modules, {'pypdfium2': None, 'fitz': None}):
            result = DocumentPerceptionAgent().perceive(str(path), image_agent=Mock())
        self.assertEqual(result.confidence, 0)
        self.assertEqual(result.facts, [])
        self.assertIn('pypdfium2', ' '.join(result.missing_info))
        self.assertIn('not installed', ' '.join(result.missing_info))

    def test_existing_pymupdf_can_render_when_pdfium_is_missing(self):
        path = self._blank_pdf()
        page = Mock()
        page.rect.width, page.rect.height = 612, 792
        pixmap = page.get_pixmap.return_value
        pixmap.width, pixmap.height = 1224, 1584
        pixmap.save.side_effect = lambda filename: Path(filename).write_bytes(b'local raster fixture')
        document = Mock()
        document.__getitem__ = Mock(return_value=page)
        renderer = types.SimpleNamespace(open=Mock(return_value=document), Matrix=Mock(), csRGB='RGB')
        agent = Mock()
        agent.perceive.return_value = PerceptionResult(
            modality='image', summary='fixture', raw_output='Backend intern Python SQL', confidence=0.6,
        )
        with patch.dict(sys.modules, {'pypdfium2': None, 'fitz': renderer}):
            result = DocumentPerceptionAgent().perceive(str(path), image_agent=agent)
        self.assertGreater(result.confidence, 0)
        self.assertTrue(result.requires_confirmation)
        self.assertIn('Python', result.raw_output)
        self.assertFalse(Path(agent.perceive.call_args.args[0]).exists())
        document.close.assert_called_once()
        self.assertFalse(page.get_pixmap.call_args.kwargs['alpha'])

    def test_scanned_result_retains_original_page_source(self):
        path = self._blank_pdf(pages=2)
        with patch.object(DocumentPerceptionAgent, '_read_scanned_page', return_value='技能：Python和SQL。') as vision:
            result = DocumentPerceptionAgent().perceive(str(path), image_agent=Mock())
        self.assertEqual(vision.call_count, 2)
        self.assertIn('[page 1]', result.raw_output)
        self.assertIn('[page 2]', result.raw_output)
        self.assertEqual(result.evidence[1].source, f'{path}#page=2')
        self.assertTrue(any('requires source verification' in item for item in result.missing_info))
        self.assertTrue(result.requires_confirmation)

    def test_pdf_page_budget_prevents_extra_model_calls(self):
        path = self._blank_pdf(pages=12)
        with patch.object(DocumentPerceptionAgent, '_read_scanned_page', return_value='可见技能：Python。') as vision:
            result = DocumentPerceptionAgent().perceive(str(path), image_agent=Mock())
        self.assertEqual(vision.call_count, 10)
        self.assertTrue(any('first 10 pages' in item for item in result.missing_info))
        self.assertNotIn('[page 11]', result.raw_output)

    def test_render_is_bounded_and_temporary_files_are_removed(self):
        path = self._blank_pdf()
        image = Mock(width=1200, height=1600, size=(1200, 1600))
        bitmap = Mock()
        bitmap.to_pil.return_value = image
        page = Mock()
        page.get_size.return_value = (1200, 1600)
        page.render.return_value = bitmap
        pdf = Mock()
        pdf.__getitem__ = Mock(return_value=page)
        module = types.SimpleNamespace(PdfDocument=Mock(return_value=pdf))
        agent = Mock()
        agent.perceive.return_value = PerceptionResult(
            modality='image', summary='模拟提取', facts=['Python'], raw_output='Python', confidence=0.6,
        )
        with patch.dict(sys.modules, {'pypdfium2': module}):
            result = DocumentPerceptionAgent().perceive(str(path), image_agent=agent)
        self.assertIn('Python', result.facts)
        self.assertLessEqual(page.render.call_args.kwargs['scale'], 1.5)
        temporary_image = Path(agent.perceive.call_args.args[0])
        self.assertFalse(temporary_image.parent.exists())
        for resource in (image, bitmap, page, pdf):
            resource.close.assert_called_once()

    def test_invalid_page_dimensions_do_not_allocate_or_call_model(self):
        path = self._blank_pdf()
        page = Mock()
        page.get_size.return_value = (float('nan'), 800)
        pdf = Mock()
        pdf.__getitem__ = Mock(return_value=page)
        module = types.SimpleNamespace(PdfDocument=Mock(return_value=pdf))
        agent = Mock()
        with patch.dict(sys.modules, {'pypdfium2': module}):
            result = DocumentPerceptionAgent().perceive(str(path), image_agent=agent)
        page.render.assert_not_called()
        agent.perceive.assert_not_called()
        self.assertEqual(result.confidence, 0)
        self.assertIn('ValueError', ' '.join(result.missing_info))

    def test_plain_pdf_text_does_not_render_or_call_vision(self):
        path = self._blank_pdf()
        fake_reader = Mock(pages=[Mock()])
        fake_reader.pages[0].extract_text.return_value = '本科大三，熟悉Python'
        image_agent = Mock()
        with patch('pypdf.PdfReader', return_value=fake_reader), patch.object(DocumentPerceptionAgent, '_read_scanned_page') as renderer:
            result = DocumentPerceptionAgent().perceive(str(path), image_agent=image_agent)
        renderer.assert_not_called()
        image_agent.perceive.assert_not_called()
        self.assertIn('Python', result.raw_output)
        self.assertFalse(result.requires_confirmation)

    def test_mixed_pdf_requires_confirmation_after_any_visual_fallback(self):
        path = self._blank_pdf(pages=2)
        fake_reader = Mock(pages=[Mock(), Mock()])
        fake_reader.pages[0].extract_text.return_value = '用户原文：本科统计学'
        fake_reader.pages[1].extract_text.return_value = ''
        with patch('pypdf.PdfReader', return_value=fake_reader), patch.object(
            DocumentPerceptionAgent, '_read_scanned_page', return_value='视觉候选：SQL',
        ):
            result = DocumentPerceptionAgent().perceive(str(path), image_agent=Mock())
        self.assertTrue(result.requires_confirmation)
        self.assertIn('本科统计学', result.raw_output)
        self.assertIn('视觉候选：SQL', result.raw_output)


class TestVideoExpansion(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / 'sample.mp4'
        self.path.write_bytes(b'mock video')

    def test_no_decodable_frames_produce_no_claimed_facts(self):
        agent = VideoPerceptionAgent()
        with patch.object(agent, '_extract_frames', return_value=[]), patch.object(agent, '_get_processor') as model:
            result = agent.perceive(str(self.path))
        model.assert_not_called()
        self.assertEqual(result.confidence, 0)
        self.assertEqual(result.facts, [])
        self.assertEqual(result.raw_output, '')
        self.assertTrue(result.missing_info)

    def test_failed_frames_are_not_facts_and_temporary_frames_are_removed(self):
        frame_dir = self.root / 'frames'
        frame_dir.mkdir()
        frames = [frame_dir / f'frame_{number}s.jpg' for number in range(2)]
        for frame in frames:
            frame.write_bytes(b'mock image')
        processor = Mock()
        processor.analyze.side_effect = ['海报要求Python。', RuntimeError('private /path and secret')]
        agent = VideoPerceptionAgent()
        with patch.object(agent, '_extract_frames', return_value=[str(item) for item in frames]), patch.object(agent, '_get_processor', return_value=processor):
            result = agent.perceive(str(self.path))
        self.assertIn('Python', result.raw_output)
        self.assertNotIn('private', str(result.model_dump()))
        self.assertTrue(any('RuntimeError' in warning for warning in result.missing_info))
        self.assertEqual(len(result.evidence), 1)
        self.assertIn('#frame=frame_0s', result.evidence[0].source)
        self.assertFalse(frame_dir.exists())
        processor.unload.assert_called_once()

    def test_all_model_failures_return_zero_confidence(self):
        frame_dir = self.root / 'frames'
        frame_dir.mkdir()
        frame = frame_dir / 'frame_0s.jpg'
        frame.write_bytes(b'mock image')
        processor = Mock()
        processor.analyze.side_effect = RuntimeError('not a career fact')
        agent = VideoPerceptionAgent()
        with patch.object(agent, '_extract_frames', return_value=[str(frame)]), patch.object(agent, '_get_processor', return_value=processor):
            result = agent.perceive(str(self.path))
        self.assertEqual(result.confidence, 0)
        self.assertEqual(result.facts, [])
        self.assertEqual(result.raw_output, '')
        self.assertFalse(frame_dir.exists())

    def test_short_video_still_samples_frame_zero(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: 30.0 if prop == 'fps' else 1
        capture.read.return_value = (True, object())

        def write_image(path, _image):
            Path(path).write_bytes(b'mock image')
            return True

        module = types.SimpleNamespace(
            VideoCapture=Mock(return_value=capture), CAP_PROP_FPS='fps',
            CAP_PROP_FRAME_COUNT='count', CAP_PROP_POS_MSEC='pos', imwrite=write_image,
        )
        processor = Mock()
        processor.analyze.return_value = '招聘要求SQL'
        agent = VideoPerceptionAgent()
        with patch.dict(sys.modules, {'cv2': module}), patch.object(agent, '_get_processor', return_value=processor):
            result = agent.perceive(str(self.path))
        self.assertIn('SQL', result.raw_output)
        capture.set.assert_called_once_with('pos', 0)
        capture.release.assert_called_once()

    def test_partial_imwrite_failures_leave_no_private_frame_files(self):
        for keep_first in (False, True):
            with self.subTest(keep_first=keep_first):
                capture = Mock()
                capture.isOpened.return_value = True
                capture.get.side_effect = lambda prop: 30.0 if prop == 'fps' else 600
                capture.read.return_value = (True, object())
                paths = []

                def write_partial(path, _image):
                    paths.append(Path(path))
                    Path(path).write_bytes(b'private partial image')
                    return keep_first and len(paths) == 1

                module = types.SimpleNamespace(
                    VideoCapture=Mock(return_value=capture), CAP_PROP_FPS='fps',
                    CAP_PROP_FRAME_COUNT='count', CAP_PROP_POS_MSEC='pos', imwrite=write_partial,
                )
                processor = Mock()
                processor.analyze.return_value = '可见Python'
                agent = VideoPerceptionAgent(max_frames=2)
                with patch.dict(sys.modules, {'cv2': module}), patch.object(agent, '_get_processor', return_value=processor):
                    result = agent.perceive(str(self.path))
                self.assertEqual(len(paths), 2)
                self.assertTrue(all(not item.exists() and not item.parent.exists() for item in paths))
                self.assertIsNone(agent._frame_directory)
                self.assertEqual(result.confidence > 0, keep_first)


if __name__ == '__main__':
    unittest.main()
