# classifier.py
import os
import re
from pathlib import Path
import json

class InputClassifier:
    """输入类型分类器 - 完全无需AI模型"""
    
    # 文件类型映射
    FILE_TYPES = {
        '图片': {
            'extensions': {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg'},
            'description': '图像文件'
        },
        '音频': {
            'extensions': {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'},
            'description': '音频文件'
        },
        '视频': {
            'extensions': {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv'},
            'description': '视频文件'
        },
        '表格': {
            'extensions': {'.xlsx', '.xls', '.csv', '.tsv', '.ods'},
            'description': '表格数据文件'
        },
        '文档': {
            'extensions': {'.txt', '.md', '.pdf', '.doc', '.docx'},
            'description': '文本文档'
        },
        '代码': {
            'extensions': {'.py', '.js', '.html', '.css', '.java', '.cpp'},
            'description': '源代码文件'
        },
        '压缩包': {
            'extensions': {'.zip', '.rar', '.7z', '.tar', '.gz'},
            'description': '压缩文件'
        }
    }
    
    # 文件路径模式
    PATH_PATTERNS = [
        re.compile(r'^[A-Za-z]:[\\/][\w\s\\/.-]+\.[a-zA-Z0-9]{2,4}$'),  # Windows
        re.compile(r'^[/~][\w\s/-]+\.[a-zA-Z0-9]{2,4}$'),               # Unix
        re.compile(r'^[\w\s-]+\.[a-zA-Z0-9]{2,4}$'),                    # 简单文件名
    ]
    
    def classify(self, input_str):
        """分类输入内容"""
        input_str = input_str.strip()
        
        # 1. 检查是否是文件路径
        if self._is_file_path(input_str):
            return self._classify_file(input_str)
        
        # 2. 检查是否是URL
        if self._is_url(input_str):
            return {'type': '链接', 'content': input_str}
        
        # 3. 否则作为文字处理
        return self._classify_text(input_str)
    
    def _is_file_path(self, text):
        """检查是否是文件路径"""
        for pattern in self.PATH_PATTERNS:
            if pattern.match(text):
                return True
        return False
    
    def _is_url(self, text):
        """检查是否是URL"""
        url_pattern = re.compile(
            r'^(https?:\/\/)?'  # http:// 或 https://
            r'([\da-z\.-]+)\.'  # 域名
            r'([a-z\.]{2,6})'   # 顶级域名
            r'([\/\w \.-]*)*'   # 路径
            r'\/?$'
        )
        return bool(url_pattern.match(text))
    
    def _classify_file(self, file_path):
        """分类文件"""
        ext = Path(file_path).suffix.lower()
        exists = os.path.exists(file_path)
        
        # 根据扩展名判断类型
        file_type = '其他'
        for type_name, type_info in self.FILE_TYPES.items():
            if ext in type_info['extensions']:
                file_type = type_name
                break
        
        result = {
            'type': file_type,
            'path': file_path,
            'exists': exists,
            'extension': ext
        }
        
        # 如果文件存在，获取更多信息
        if exists:
            result['size'] = os.path.getsize(file_path)
            result['size_kb'] = round(result['size'] / 1024, 2)
        
        return result
    
    def _classify_text(self, text):
        """分类文字"""
        # 检查是否是JSON
        try:
            json.loads(text)
            return {'type': '文字', 'subtype': 'JSON', 'content_preview': text[:100]}
        except:
            pass
        
        # 检查是否是CSV
        if ',' in text and '\n' in text:
            return {'type': '文字', 'subtype': 'CSV', 'content_preview': text[:100]}
        
        # 普通文字
        return {
            'type': '文字',
            'subtype': '普通文本',
            'length': len(text),
            'words': len(text.split()),
            'content_preview': text[:100]
        }
    
    def print_result(self, result):
        """友好打印结果"""
        if result['type'] == '文字':
            print(f"📝 文字内容")
            print(f"   类型: {result.get('subtype', '普通文本')}")
            if 'length' in result:
                print(f"   长度: {result['length']} 字符")
            if 'content_preview' in result:
                print(f"   预览: {result['content_preview']}")
        
        elif result['type'] == '链接':
            print(f"🔗 网页链接")
            print(f"   地址: {result['content']}")
        
        else:
            # 文件类型
            emoji = {
                '图片': '🖼️',
                '音频': '🎵',
                '视频': '🎬',
                '表格': '📊',
                '文档': '📄',
                '代码': '💻',
                '压缩包': '🗜️' #可删去？或增加解压功能？
            }.get(result['type'], '📁')
            
            print(f"{emoji} {result['type']}文件")
            print(f"   路径: {result['path']}")
            print(f"   存在: {'✓' if result['exists'] else '✗'}")
            if result.get('size_kb'):
                print(f"   大小: {result['size_kb']} KB")

def main():
    """主函数 - 交互式使用"""
    classifier = InputClassifier()
    
    print("=" * 50)
    print("🔍 输入类型分类器 (无需AI模型)")
    print("=" * 50)
    print("输入任意内容，我会告诉你它是什么类型")
    print("输入 'quit' 退出程序")
    print()
    
    while True:
        # 获取用户输入
        user_input = input("请输入 > ").strip()
        
        # 检查是否退出
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 再见！")
            break
        
        if not user_input:
            continue
        
        # 分类
        print("\n🤔 分析中...")
        result = classifier.classify(user_input)
        
        # 显示结果
        print("\n✅ 分类结果:")
        classifier.print_result(result)
        print("-" * 40)

if __name__ == "__main__":
    main()