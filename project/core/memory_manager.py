# memory_manager.py
import torch
import gc
import time
from contextlib import contextmanager


class VRAMManager:
    """显存管理器 - 严格管理6GB显存下的模型切换"""
    
    def __init__(self, max_memory_gb=6.0):
        self.max_memory = max_memory_gb * 1024**3
        self.loaded_models = {}
        self.model_sizes = {
            'text': 2.5 * 1024**3,
            'vision': 4.0 * 1024**3,
        }
        print(f"🎮 显存管理器初始化 | 限制: {max_memory_gb}GB")
        self._print_gpu_info()
        self._initial_cleanup()
    
    def _initial_cleanup(self):
        """初始清理"""
        print("🧹 初始显存清理...")
        self._hard_cleanup()
    
    def _print_gpu_info(self):
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   GPU: {gpu_name} | 总显存: {total_memory:.1f}GB")
            print(f"   ⚠️  严格限制: 6.0GB（保留2GB系统余量）")
    
    def _get_memory_info(self):
        """获取详细显存信息"""
        if not torch.cuda.is_available():
            return None, None
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        return allocated, reserved
    
    def _hard_cleanup(self):
        """强制深度清理"""
        if not torch.cuda.is_available():
            return
        
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    def unload_model(self, model_type):
        """彻底卸载模型"""
        if model_type not in self.loaded_models:
            return
        
        print(f"🗑️  正在深度卸载 {model_type} 模型...")
        
        model_data = self.loaded_models[model_type]
        model = model_data.get('model')
        processor = model_data.get('processor')
        tokenizer = model_data.get('tokenizer')
        
        torch.cuda.synchronize()
        
        if hasattr(model, 'cpu'):
            try:
                model.cpu()
            except:
                pass
        torch.cuda.synchronize()
        
        del model
        if processor is not None:
            del processor
        if tokenizer is not None:
            del tokenizer
            
        del self.loaded_models[model_type]
        
        self._hard_cleanup()
        
        allocated, reserved = self._get_memory_info()
        print(f"   ✅ 卸载完成 | 当前分配: {allocated:.2f}GB | 预留: {reserved:.2f}GB")
    
    def unload_all(self):
        """卸载所有模型"""
        print("🧹 卸载所有模型...")
        for model_type in list(self.loaded_models.keys()):
            self.unload_model(model_type)
        self._hard_cleanup()
        print("✅ 显存已完全释放")
    
    def register_model(self, model_type, model, processor=None, tokenizer=None):
        """注册模型及其组件 - 使用明确的类型"""
        self.loaded_models[model_type] = {
            'model': model,
            'processor': processor,
            'tokenizer': tokenizer
        }
        allocated, reserved = self._get_memory_info()
        print(f"📥 已注册 {model_type} | 分配: {allocated:.2f}GB | 预留: {reserved:.2f}GB")
    
    def ensure_loaded(self, model_type, load_func):
        """确保模型加载，显存不足时卸载其他模型"""
        if model_type in self.loaded_models:
            print(f"✅ {model_type} 已在显存中")
            return self.loaded_models[model_type]['model']
        
        allocated, _ = self._get_memory_info()
        print(f"💾 当前显存分配: {allocated:.2f}GB")
        
        # 卸载其他所有模型
        for other_type in list(self.loaded_models.keys()):
            if other_type != model_type:
                self.unload_model(other_type)
        
        self._hard_cleanup()
        
        allocated, _ = self._get_memory_info()
        print(f"💾 清理后显存: {allocated:.2f}GB")
        
        # 加载新模型
        print(f"⏳ 正在加载 {model_type}...")
        start_time = time.time()
        
        try:
            # load_func 现在返回 (model, processor, tokenizer)
            model, processor, tokenizer = load_func()
            load_time = time.time() - start_time
            
            self.register_model(model_type, model, processor, tokenizer)
            print(f"✅ 加载完成，耗时: {load_time:.1f}s")
            
            allocated, reserved = self._get_memory_info()
            print(f"💾 加载后显存 - 分配: {allocated:.2f}GB | 预留: {reserved:.2f}GB")
            
            if allocated > 5.5:
                print(f"   ⚠️  警告: 显存使用接近限制 ({allocated:.2f}GB/6.0GB)")
            
            return model
            
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            self._hard_cleanup()
            raise
    
    def get_status(self):
        """获取显存状态"""
        allocated, reserved = self._get_memory_info()
        loaded = list(self.loaded_models.keys())
        return f"分配: {allocated:.2f}GB | 预留: {reserved:.2f}GB | 模型: {loaded}"


# 全局实例
_vram_manager = None

def get_vram_manager():
    global _vram_manager
    if _vram_manager is None:
        _vram_manager = VRAMManager()
    return _vram_manager

def cleanup_all():
    global _vram_manager
    if _vram_manager:
        _vram_manager.unload_all()
        _vram_manager = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()