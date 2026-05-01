# memory_manager.py
import torch
import gc
import time
from contextlib import contextmanager


class VRAMManager:
    """VRAM Manager - strict 6GB model switching"""

    def __init__(self, max_memory_gb=6.0):
        self.max_memory = max_memory_gb * 1024**3
        self.loaded_models = {}
        self.model_sizes = {
            'vision': 4.0 * 1024**3,
        }
        print(f"[VRAM] Manager init | Max: {max_memory_gb}GB")
        self._print_gpu_info()
        self._initial_cleanup()

    def _initial_cleanup(self):
        print("[VRAM] Initial cleanup...")
        self._hard_cleanup()

    def _print_gpu_info(self):
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   GPU: {gpu_name} | Total VRAM: {total_memory:.1f}GB")
            print(f"   [WARN] Strict limit: 6.0GB (2GB reserved for system)")

    def _get_memory_info(self):
        if not torch.cuda.is_available():
            return None, None
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        return allocated, reserved

    def _hard_cleanup(self):
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
        if model_type not in self.loaded_models:
            return

        print(f"[UNLOAD] Deep unloading {model_type} model...")

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
        print(f"   [OK] Unload complete | Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB")

    def unload_all(self):
        print("[UNLOAD] Unloading all models...")
        for model_type in list(self.loaded_models.keys()):
            self.unload_model(model_type)
        self._hard_cleanup()
        print("[OK] VRAM fully released")

    def register_model(self, model_type, model, processor=None, tokenizer=None):
        self.loaded_models[model_type] = {
            'model': model,
            'processor': processor,
            'tokenizer': tokenizer
        }
        allocated, reserved = self._get_memory_info()
        print(f"[REGISTER] {model_type} | Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB")

    def ensure_loaded(self, model_type, load_func):
        if model_type in self.loaded_models:
            print(f"[OK] {model_type} already in VRAM")
            return self.loaded_models[model_type]['model']

        allocated, _ = self._get_memory_info()
        print(f"[VRAM] Current allocated: {allocated:.2f}GB")

        for other_type in list(self.loaded_models.keys()):
            if other_type != model_type:
                self.unload_model(other_type)

        self._hard_cleanup()

        allocated, _ = self._get_memory_info()
        print(f"[VRAM] After cleanup: {allocated:.2f}GB")

        print(f"[LOAD] Loading {model_type}...")
        start_time = time.time()

        try:
            model, processor, tokenizer = load_func()
            load_time = time.time() - start_time

            self.register_model(model_type, model, processor, tokenizer)
            print(f"[OK] Load complete, time: {load_time:.1f}s")

            allocated, reserved = self._get_memory_info()
            print(f"[VRAM] After load - Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB")

            if allocated > 5.5:
                print(f"   [WARN] VRAM usage near limit ({allocated:.2f}GB/6.0GB)")

            return model

        except Exception as e:
            print(f"[ERROR] Load failed: {e}")
            self._hard_cleanup()
            raise

    def get_status(self):
        allocated, reserved = self._get_memory_info()
        loaded = list(self.loaded_models.keys())
        return f"Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Models: {loaded}"


# Global instance
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
