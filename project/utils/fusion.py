# fusion_processor.py（保持不变）
import re


class FusionProcessor:
    def __init__(self, text_processor):
        self.text_processor = text_processor
    
    def fuse(self, text_query, image_results, stream=False):
        fusion_prompt = self._build_fusion_prompt(text_query, image_results)
        
        system_prompt = """你是一个多模态AI助手。你将收到：
1. 用户的原始问题
2. 一张或多张图片的内容描述

你的任务是：
- 结合图片信息和用户问题，提供全面、准确的回答
- 如果图片与问题直接相关，重点分析图片内容回答问题
- 如果图片是辅助信息，将其作为上下文参考
- 保持回答的连贯性和逻辑性"""
        
        return self.text_processor.generate(
            query=fusion_prompt,
            system_prompt=system_prompt,
            stream=stream,
            max_tokens=2048
        )
    
    def _build_fusion_prompt(self, text_query, image_results):
        prompt_parts = ["【用户问题】", text_query, "", "【图片分析】"]
        
        for i, result in enumerate(image_results, 1):
            prompt_parts.append(f"图片{i} ({result['path']}):")
            prompt_parts.append(result['description'])
            prompt_parts.append("")
        
        prompt_parts.extend([
            "【任务】",
            "请结合以上图片分析内容，回答用户的问题。如果图片包含与问题相关的关键信息，请重点引用；如果问题是开放式的，请综合图片内容提供见解。"
        ])
        
        return "\n".join(prompt_parts)
    
    def simple_combine(self, text_response, image_descriptions):
        output = ["=" * 50, "📝 文本分析结果", "=" * 50, "", text_response, ""]
        
        if image_descriptions:
            output.extend(["=" * 50, "🖼️ 图片分析结果", "=" * 50, ""])
            for i, desc in enumerate(image_descriptions, 1):
                output.append(f"【图片{i}】")
                output.append(desc)
                output.append("")
        
        output.append("=" * 50)
        return "\n".join(output)