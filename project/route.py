"""
输入分类模块 - 支持文本、图片、图文混合输入
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple


class InputClassifier:
    """输入类型分类器 - 支持多模态输入检测"""
    
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg'}
    
    # 路径匹配模式
    PATH_PATTERNS = [
        re.compile(r'^[A-Za-z]:[\\/][\w\s\\/.-]+\.[a-zA-Z0-9]{2,4}$'),  # Windows绝对
        re.compile(r'^[/~][\w\s/-]+\.[a-zA-Z0-9]{2,4}$'),               # Unix绝对
        re.compile(r'^[\w\s-]+\.[a-zA-Z0-9]{2,4}$'),                    # 相对路径
    ]
    
    def classify(self, input_str: str) -> Dict:
        """
        分类输入内容，支持：
        - 纯文本
        - 单张/多张图片路径
        - 图文混合（文本中包含图片路径）
        
        Returns:
            {
                'mode': 'text' | 'image' | 'multimodal',
                'text_content': str,  # 提取的文本部分
                'image_paths': List[str],  # 提取的图片路径
                'raw_input': str
            }
        """
        input_str = input_str.strip()
        
        # 查找所有可能的图片路径
        image_paths = self._extract_image_paths(input_str)
        
        # 提取纯文本部分（移除图片路径）
        text_content = self._extract_text_content(input_str, image_paths)
        
        # 判断模式
        if image_paths and text_content:
            mode = 'multimodal'
        elif image_paths:
            mode = 'image'
        else:
            mode = 'text'
        
        # 验证图片是否存在
        valid_images = []
        for path in image_paths:
            valid_images.append({
                'path': path,
                'exists': os.path.exists(path),
                'absolute': os.path.abspath(path) if os.path.exists(path) else path
            })
        
        return {
            'mode': mode,
            'text_content': text_content,
            'image_paths': [img['path'] for img in valid_images],
            'image_details': valid_images,
            'raw_input': input_str
        }
    
    def _extract_image_paths(self, text: str) -> List[str]:
        """从文本中提取所有图片路径"""
        # 按空格、换行、逗号、分号分割
        delimiters = r'[\s\n,;]+'
        tokens = re.split(delimiters, text)
        
        image_paths = []
        for token in tokens:
            token = token.strip().strip('"\'')  # 去除引号
            if self._is_image_path(token):
                image_paths.append(token)
        
        return image_paths
    
    def _is_image_path(self, text: str) -> bool:
        """检查是否是图片路径"""
        if not text or len(text) < 4:
            return False
        
        # 检查是否符合路径模式
        is_path = any(pattern.match(text) for pattern in self.PATH_PATTERNS)
        if not is_path:
            return False
        
        ext = Path(text).suffix.lower()
        return ext in self.IMAGE_EXTENSIONS
    
    def _extract_text_content(self, raw_input: str, image_paths: List[str]) -> str:
        """提取纯文本内容（移除图片路径）"""
        text = raw_input
        
        # 移除所有图片路径
        for path in image_paths:
            # 处理各种可能的包围字符
            for wrapper in [f' {path} ', f'"{path}"', f"'{path}'", 
                           f',{path},', f';{path};', f'\n{path}\n',
                           f'{path} ', f' {path}', f'{path},', f'{path};']:
                text = text.replace(wrapper, ' ')
            text = text.replace(path, ' ')
        
        # 清理多余空白
        text = ' '.join(text.split())
        return text.strip()
    
    def validate(self, classification: Dict) -> Tuple[bool, str]:
        """验证分类结果"""
        if classification['mode'] == 'image':
            # 检查至少有一张图片存在
            existing = [img for img in classification['image_details'] if img['exists']]
            if not existing:
                return False, "未找到有效的图片文件"
        
        elif classification['mode'] == 'multimodal':
            # 检查文本非空且至少有一张图片存在
            if not classification['text_content']:
                return False, "图文模式下文本内容不能为空"
            existing = [img for img in classification['image_details'] if img['exists']]
            if not existing:
                return False, "未找到有效的图片文件"
        
        return True, "验证通过"


def classify_input(input_str: str) -> Dict:
    """便捷函数"""
    classifier = InputClassifier()
    return classifier.classify(input_str)