#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List

from .core.input_router import DataRouter
from .core.memory_manager import cleanup_all, get_vram_manager


class MultimodalAssistant:
    def __init__(
        self,
        image_model_path: str = "./models/Qwen3-VL-2B-Instruct",
    ) -> None:
        print("Initializing multimodal assistant...")
        print("=" * 60)

        self.router = DataRouter()
        self.vram_manager = get_vram_manager()

        try:
            from .agents.image import ImageProcessor
        except Exception as e:
            raise RuntimeError(
                "Failed to import model processors. "
                "Please check transformers/huggingface-hub compatibility."
            ) from e

        self.image_processor = ImageProcessor(model_path=image_model_path)

        # Cloud brain client for text generation (lazy init)
        self._brain_client = None

        # Delegate document/audio processing to the shared pipeline
        self._pipeline = None

        print("=" * 60)
        print("System ready | VRAM manager: enabled")
        print("Supported modes: text | image | multimodal | file | audio_video")
        print("Text mode uses cloud DeepSeek API (streaming)")
        print("-" * 60)

    def _get_brain_client(self):
        if self._brain_client is None:
            from .core.brain_client import DeepSeekBrainClient

            self._brain_client = DeepSeekBrainClient()
        return self._brain_client

    def _get_pipeline(self):
        if self._pipeline is None:
            from .core.multimodal_pipeline import MultimodalChatPipeline

            self._pipeline = MultimodalChatPipeline(
                image_model_path="./models/Qwen3-VL-2B-Instruct",
            )
        return self._pipeline

    def process(self, user_input: str, stream: bool = False) -> Dict[str, Any]:
        print("\nAnalyzing input...")
        routed = self.router.route(user_input)

        if routed.get("error"):
            return {"success": False, "error": routed["error"]}

        mode = routed["route"]
        metadata = routed["metadata"]

        print(f"Detected mode: {mode}")
        print(f"Text length: {len(metadata.get('text_content', ''))}")
        print(f"Image count: {len(metadata.get('image_paths', []))}")
        print(f"File count: {len(metadata.get('file_paths', []))}")
        print(f"Audio/Video count: {len(metadata.get('audio_video_paths', []))}")
        print(f"Target agent: {routed['target_agent']}")

        valid, msg = self.router.classifier.validate(metadata)
        if not valid:
            print(f"Validation failed: {msg}")
            return {"success": False, "mode": mode, "error": msg}

        try:
            if mode == "text":
                return self._handle_text_only(metadata, stream)
            if mode == "image":
                return self._handle_image_only(metadata)
            if mode == "multimodal":
                return self._handle_multimodal(metadata)
            if mode == "file":
                return self._handle_file(metadata, stream)
            if mode == "audio_video":
                return self._handle_audio_video(metadata, stream)
            return {"success": False, "mode": mode, "error": f"unknown mode: {mode}"}
        except Exception as e:
            print(f"Processing error: {e}")
            return {"success": False, "mode": mode, "error": str(e)}
        finally:
            print(f"\n{self.vram_manager.get_status()}")

    def _handle_text_only(self, classification: Dict[str, Any], stream: bool) -> Dict[str, Any]:
        query = classification["text_content"]
        print("\nHandling text query with cloud DeepSeek API...")

        if stream:
            return self._stream_cloud_output(query)

        brain = self._get_brain_client()
        response = brain.plan(query)
        print("\n" + "=" * 60)
        print(response)
        print("=" * 60)
        return {
            "success": True,
            "mode": "text",
            "response": response,
            "model": brain.model_name,
        }

    def _handle_image_only(self, classification: Dict[str, Any]) -> Dict[str, Any]:
        images = classification["image_details"]
        valid_images = [img for img in images if img["exists"]]
        print("\nHandling image input with Qwen-VL...")

        results: List[Dict[str, str]] = []
        for img_info in valid_images:
            print(f"  analyzing: {img_info['path']}")
            desc = self.image_processor.analyze(img_info["path"])
            results.append({"path": img_info["path"], "description": desc})
            print(f"  done: {img_info['path']}")

        print("\n" + "=" * 60)
        print("Qwen-VL results:")
        print("=" * 60)
        for i, result in enumerate(results, 1):
            if len(results) > 1:
                print(f"\n[Image {i}] {result['path']}")
                print("-" * 40)
            print(result["description"])
        print("=" * 60)

        self.image_processor.unload()
        return {
            "success": True,
            "mode": "image",
            "results": results,
            "model": "Qwen-VL",
        }

    def _handle_multimodal(self, classification: Dict[str, Any]) -> Dict[str, Any]:
        text_query = classification["text_content"]
        images = [img for img in classification["image_details"] if img["exists"]]

        print("\nHandling multimodal input with Qwen-VL...")
        print(f"Question: {text_query}")

        results: List[Dict[str, str]] = []
        for img_info in images:
            print(f"  analyzing: {img_info['path']}")
            desc = self.image_processor.analyze(img_info["path"], question=text_query)
            results.append({"path": img_info["path"], "description": desc})
            print(f"  done: {img_info['path']}")

        print("\n" + "=" * 60)
        print("Qwen-VL results:")
        print("=" * 60)
        for i, result in enumerate(results, 1):
            if len(results) > 1:
                print(f"\n[Image {i}] {result['path']}")
                print("-" * 40)
            print(result["description"])
        print("=" * 60)

        self.image_processor.unload()
        return {
            "success": True,
            "mode": "multimodal",
            "query": text_query,
            "results": results,
            "model": "Qwen-VL",
        }

    def _handle_file(self, classification: Dict[str, Any], stream: bool) -> Dict[str, Any]:
        file_paths = classification["file_paths"]
        text_context = classification.get("text_content", "")
        print(f"\nHandling document(s): {file_paths}")

        pipeline = self._get_pipeline()
        content = pipeline._understand_documents(file_paths, text_context)
        print(f"\nDocument extraction complete ({len(content)} chars)")
        print("-" * 60)
        print(content[:2000] if len(content) > 2000 else content)
        print("-" * 60)

        if text_context and stream:
            query = f"Based on the following document content, {text_context}\n\nDocument:\n{content}"
            return self._stream_cloud_output(query)

        return {
            "success": True,
            "mode": "file",
            "file_paths": file_paths,
            "extracted_content": content,
            "model": "DocumentParser",
        }

    def _handle_audio_video(self, classification: Dict[str, Any], stream: bool) -> Dict[str, Any]:
        media_paths = classification["audio_video_paths"]
        text_context = classification.get("text_content", "")
        print(f"\nHandling audio/video: {media_paths}")

        pipeline = self._get_pipeline()
        content = pipeline._understand_audio_video(media_paths, text_context)
        print(f"\nAudio/video extraction complete ({len(content)} chars)")
        print("-" * 60)
        print(content[:2000] if len(content) > 2000 else content)
        print("-" * 60)

        if text_context and stream:
            query = f"Based on the following audio/video content, {text_context}\n\nContent:\n{content}"
            return self._stream_cloud_output(query)

        return {
            "success": True,
            "mode": "audio_video",
            "media_paths": media_paths,
            "extracted_content": content,
            "model": "AudioProcessor",
        }

    def _stream_cloud_output(self, query: str) -> Dict[str, Any]:
        print("\n" + "=" * 60)
        print("Cloud DeepSeek streaming...")
        print("-" * 60)

        brain = self._get_brain_client()
        full_response = ""
        try:
            for token in brain.plan_stream(query):
                print(token, end="", flush=True)
                full_response += token
        except RuntimeError as e:
            print(f"\nError: {e}")
            return {"success": False, "mode": "text", "error": str(e)}

        print("\n" + "=" * 60)
        return {
            "success": True,
            "mode": "text",
            "response": full_response,
            "model": brain.model_name,
        }

    def interactive_mode(self) -> None:
        print("\nInteractive mode")
        print("Examples:")
        print('  "What is machine learning?"               -> text (cloud DeepSeek)')
        print('  "./photo.jpg"                             -> image (Qwen-VL)')
        print('  "Analyze this image ./img.jpg"            -> multimodal (Qwen-VL)')
        print('  "./report.pdf"                            -> file (document parser)')
        print('  "Summarize ./report.pdf"                  -> file + text query')
        print('  "./audio.mp3"                             -> audio (ASR)')
        print('  "quit" or "exit"                          -> quit')
        print("-" * 60 + "\n")

        while True:
            try:
                user_input = input("input > ").strip()
                if not user_input:
                    continue
                if user_input.lower() in {"quit", "exit", "q"}:
                    print("Cleaning resources and exiting...")
                    cleanup_all()
                    print("Exited safely.")
                    break
                self.process(user_input, stream=True)
                print()
            except KeyboardInterrupt:
                print("\nGoodbye.")
                cleanup_all()
                break
            except Exception as e:
                print(f"Error: {e}")
                continue


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal assistant entrypoint")
    parser.add_argument("input", nargs="?", help="input content (supports text/image/multimodal/file/audio_video)")
    parser.add_argument("--image-model", default="./models/Qwen3-VL-2B-Instruct")
    args = parser.parse_args()

    assistant = MultimodalAssistant(
        image_model_path=args.image_model,
    )

    if args.input:
        result = assistant.process(args.input, stream=True)
        cleanup_all()
        sys.exit(0 if result.get("success") else 1)

    try:
        assistant.interactive_mode()
    finally:
        cleanup_all()


if __name__ == "__main__":
    main()
