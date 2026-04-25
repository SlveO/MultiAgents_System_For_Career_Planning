# main.py
#!/usr/bin/env python3
import sys
import argparse
from .core.memory_manager import get_vram_manager, cleanup_all
from .classifier.classifier import InputClassifier
from .agents.text import TextProcessor
from .agents.image import ImageProcessor


class MultimodalAssistant:
    def __init__(self, 
                 text_model_path='./models/DeepSeek-R1-Distill-Qwen-1.5B',
                 image_model_path='./models/Qwen3-VL-2B-Instruct'):
        
        print("🚀 初始化多模态AI助手...")
        print("=" * 60)
        
        self.classifier = InputClassifier()
        self.vram_manager = get_vram_manager()
        
        self.text_processor = TextProcessor(model_path=text_model_path)
        self.image_processor = ImageProcessor(model_path=image_model_path)
        
        print("=" * 60)
        print("✅ 系统就绪 | 显存管理: 启用")
        print("💡 支持格式: 文本 | 图片路径 | 图文混合")
        print("   图片模式: 仅使用Vision模型直接输出")
        print("-" * 60)
    
    def process(self, user_input: str, stream: bool = False):
        print(f"\n🔍 分析输入...")
        
        classification = self.classifier.classify(user_input)
        mode = classification['mode']
        
        print(f"📋 检测模式: {self._mode_name(mode)}")
        print(f"   文本长度: {len(classification['text_content'])} 字符")
        print(f"   图片数量: {len(classification['image_paths'])} 张")
        
        valid, msg = self.classifier.validate(classification)
        if not valid:
            print(f"❌ 验证失败: {msg}")
            return {'success': False, 'error': msg}
        
        try:
            if mode == 'text':
                return self._handle_text_only(classification, stream)
            elif mode == 'image':
                return self._handle_image_only(classification)
            else:
                return self._handle_multimodal(classification)
                
        except Exception as e:
            print(f"❌ 处理出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
        
        finally:
            print(f"\n💾 {self.vram_manager.get_status()}")
    
    def _mode_name(self, mode: str) -> str:
        return {'text': '纯文本', 'image': '纯图片', 'multimodal': '图文混合'}.get(mode, '未知')
    
    def _handle_text_only(self, classification, stream):
        """纯文本模式 - 使用Text模型"""
        query = classification['text_content']
        print(f"\n💬 处理文本查询...")
        
        if stream:
            return self._stream_text_output(query)
        else:
            result = self.text_processor.generate(query, stream=False)
            self._print_text_result(result)
            return {
                'success': True,
                'mode': 'text',
                'response': result['response'],
                'thinking': result['thinking']
            }
    
    def _handle_image_only(self, classification):
        """纯图片模式 - 仅使用Vision模型直接输出"""
        images = classification['image_details']
        valid_images = [img for img in images if img['exists']]
        
        print(f"\n🖼️ 分析图片（Vision模型直接输出）...")
        
        results = []
        for img_info in valid_images:
            print(f"   🖼️  分析: {img_info['path']}")
            
            # 直接调用vision模型，不经过text模型
            desc = self.image_processor.analyze(img_info['path'])
            
            results.append({
                'path': img_info['path'],
                'description': desc
            })
            print(f"   ✅ 完成: {img_info['path']}")
        
        # 输出结果
        print("\n" + "=" * 60)
        print("📝 Vision模型分析结果:")
        print("=" * 60)
        
        for i, result in enumerate(results, 1):
            if len(results) > 1:
                print(f"\n【图片 {i}】{result['path']}")
                print("-" * 40)
            print(result['description'])
        
        print("=" * 60)
        
        # 处理完成后卸载vision模型释放显存
        self.image_processor.unload()
        
        return {
            'success': True,
            'mode': 'image',
            'results': results
        }
    
    def _handle_multimodal(self, classification):
        """图文混合模式 - 仅使用Vision模型直接输出"""
        text_query = classification['text_content']
        images = [img for img in classification['image_details'] if img['exists']]
        
        print(f"\n🖼️ 分析图文（Vision模型直接输出）...")
        print(f"   用户问题: {text_query}")
        
        results = []
        for img_info in images:
            print(f"   🖼️  分析: {img_info['path']}")
            
            # 将用户问题作为上下文传递给vision模型
            desc = self.image_processor.analyze(
                img_info['path'],
                question=text_query  # 直接将用户问题作为问题输入
            )
            
            results.append({
                'path': img_info['path'],
                'description': desc
            })
            print(f"   ✅ 完成: {img_info['path']}")
        
        # 输出结果
        print("\n" + "=" * 60)
        print("📝 Vision模型分析结果:")
        print("=" * 60)
        
        for i, result in enumerate(results, 1):
            if len(results) > 1:
                print(f"\n【图片 {i}】{result['path']}")
                print("-" * 40)
            print(result['description'])
        
        print("=" * 60)
        
        # 处理完成后卸载vision模型释放显存
        self.image_processor.unload()
        
        return {
            'success': True,
            'mode': 'multimodal',
            'query': text_query,
            'results': results
        }
    
    def _stream_text_output(self, query):
        """流式文本输出"""
        print("\n" + "=" * 60)
        print("🤔 思考中...")
        print("-" * 60)
        
        full_response = ""
        thinking = ""
        
        for chunk in self.text_processor.generate(query, stream=True):
            if chunk['type'] == 'thinking':
                print(chunk['token'], end='', flush=True)
            elif chunk['type'] == 'thinking_complete':
                print("\n" + "-" * 60)
                print("💡 回答:")
                print(chunk['token'], end='', flush=True)
                thinking = chunk['thinking']
                full_response = chunk['token']
            elif chunk['type'] == 'response':
                print(chunk['token'], end='', flush=True)
                full_response += chunk['token']
            elif chunk['type'] == 'complete':
                thinking = chunk['thinking']
                full_response = chunk['response']
        
        print("\n" + "=" * 60)
        return {
            'success': True,
            'mode': 'text',
            'response': full_response,
            'thinking': thinking
        }
    
    def _print_text_result(self, result):
        """打印文本结果"""
        print("\n" + "=" * 60)
        if result.get('thinking'):
            print("🤔 思考过程:")
            print(result['thinking'][:300] + "..." if len(result['thinking']) > 300 else result['thinking'])
            print("-" * 60)
        print("💡 回答:")
        print(result['response'])
        print("=" * 60)
    
    def interactive_mode(self):
        print("\n🎯 进入交互模式")
        print("输入示例：")
        print('  "什么是深度学习？"                    → 纯文本（Text模型）')
        print('  "./photo.jpg"                         → 纯图片（Vision模型）')
        print('  "分析这张图片 ./img.jpg 的内容"       → 图文混合（Vision模型）')
        print('  "quit" 或 "exit"                     → 退出')
        print("-" * 60 + "\n")
        
        while True:
            try:
                user_input = input("👤 输入 > ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q', '退出']:
                    print("👋 正在清理资源并退出...")
                    cleanup_all()
                    print("✅ 已安全退出")
                    break
                
                self.process(user_input, stream=True)
                print()
                
            except KeyboardInterrupt:
                print("\n👋 再见！")
                cleanup_all()
                break
            except Exception as e:
                print(f"❌ 错误: {e}")
                continue


def main():
    parser = argparse.ArgumentParser(description='多模态AI助手 - 6GB显存优化版')
    parser.add_argument('input', nargs='?', help='输入内容（支持图文混合）')
    parser.add_argument('--text-model', default='./models/DeepSeek-R1-Distill-Qwen-1.5B')
    parser.add_argument('--image-model', default='./models/Qwen3-VL-2B-Instruct')
    
    args = parser.parse_args()
    
    assistant = MultimodalAssistant(
        text_model_path=args.text_model,
        image_model_path=args.image_model
    )
    
    if args.input:
        result = assistant.process(args.input, stream=True)
        cleanup_all()
        sys.exit(0 if result.get('success') else 1)
    else:
        try:
            assistant.interactive_mode()
        finally:
            cleanup_all()


if __name__ == "__main__":
    main()