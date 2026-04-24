import torch
import pynvml
from typing import Dict, Optional
import threading
import time

class GPUMonitor:
    """RTX4060 显存监控器（单例模式）"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # RTX4060 通常是 GPU 0
        self.peak_memory = 0
        self.monitoring = False
        
    def get_memory_info(self) -> Dict[str, float]:
        """获取当前显存使用情况（MB）"""
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        total = info.total / 1024**2
        used = info.used / 1024**2
        free = info.free / 1024**2
        
        self.peak_memory = max(self.peak_memory, used)
        
        return {
            "total_mb": round(total, 2),      # 8192.00 MB for RTX4060
            "used_mb": round(used, 2),
            "free_mb": round(free, 2),
            "usage_percent": round(used/total*100, 2),
            "peak_mb": round(self.peak_memory, 2)
        }
    
    def print_status(self, label: str = ""):
        """打印当前显存状态"""
        mem = self.get_memory_info()
        print(f"[GPU] {label} | 已用: {mem['used_mb']:.0f}MB / {mem['total_mb']:.0f}MB "
              f"({mem['usage_percent']:.1f}%) | 剩余: {mem['free_mb']:.0f}MB")
        return mem
    
    def clear_cache(self):
        """清理PyTorch显存缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        time.sleep(0.5)  # 给显存回收留出时间
    
    def start_monitoring(self, interval: float = 1.0):
        """后台监控线程"""
        self.monitoring = True
        def monitor():
            while self.monitoring:
                self.print_status("后台监控")
                time.sleep(interval)
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    
    def stop_monitoring(self):
        self.monitoring = False

# 全局监控器实例
gpu_monitor = GPUMonitor()