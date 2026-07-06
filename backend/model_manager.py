"""
统一模型管理器 - 管理 AI 模型的加载、缓存和释放
支持内存监控、自动释放、模型生命周期管理
"""
import torch
import psutil
import time
import threading
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ModelPriority(Enum):
    """模型优先级"""
    HIGH = "high"      # 高优先级，不自动释放
    MEDIUM = "medium"  # 中优先级，较少释放
    LOW = "low"        # 低优先级，优先释放

@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    model_type: str
    loaded_at: float
    last_used: float
    memory_estimate: int  # bytes
    priority: ModelPriority
    usage_count: int = 0

class ModelManager:
    """统一模型管理器"""
    
    def __init__(
        self,
        max_memory_gb: float = 4.0,
        warning_threshold: float = 0.8,
        critical_threshold: float = 0.9,
        auto_release_enabled: bool = True,
        release_interval: int = 300  # 5分钟检查一次
    ):
        """
        初始化模型管理器
        
        Args:
            max_memory_gb: 最大内存限制 (GB)
            warning_threshold: 警告阈值 (百分比)
            critical_threshold: 临界阈值 (百分比)
            auto_release_enabled: 是否启用自动释放
            release_interval: 自动释放检查间隔 (秒)
        """
        self._models: Dict[str, Any] = {}
        self._model_info: Dict[str, ModelInfo] = {}
        self._lock = threading.Lock()
        
        # 内存配置
        self._max_memory = max_memory_gb * 1024 * 1024 * 1024  # GB to bytes
        self._warning_threshold = warning_threshold
        self._critical_threshold = critical_threshold
        
        # 自动释放配置
        self._auto_release_enabled = auto_release_enabled
        self._release_interval = release_interval
        self._release_thread = None
        self._running = False
        
        # 统计信息
        self._total_loads = 0
        self._total_releases = 0
        self._peak_memory = 0
        
        logger.info(f"ModelManager initialized: max_memory={max_memory_gb}GB, "
                   f"warning_threshold={warning_threshold}, "
                   f"critical_threshold={critical_threshold}")
    
    def load_model(
        self,
        model_type: str,
        model_name: str,
        loader_func: Callable,
        priority: ModelPriority = ModelPriority.MEDIUM,
        force_reload: bool = False
    ) -> Any:
        """
        加载模型（带内存检查和缓存）
        
        Args:
            model_type: 模型类型 (whisper, demucs)
            model_name: 模型名称
            loader_func: 模型加载函数
            priority: 模型优先级
            force_reload: 是否强制重新加载
        
        Returns:
            模型实例
        """
        cache_key = f"{model_type}_{model_name}"
        
        with self._lock:
            # 检查是否已加载
            if cache_key in self._models and not force_reload:
                # 更新使用信息
                self._model_info[cache_key].last_used = time.time()
                self._model_info[cache_key].usage_count += 1
                
                logger.debug(f"Model already loaded: {cache_key}, "
                           f"usage_count={self._model_info[cache_key].usage_count}")
                
                return self._models[cache_key]
            
            # 检查内存使用
            current_memory = psutil.Process().memory_info().rss
            
            # 更新峰值内存
            if current_memory > self._peak_memory:
                self._peak_memory = current_memory
            
            # 如果超过临界阈值，紧急释放
            if current_memory > self._max_memory * self._critical_threshold:
                logger.warning(f"Memory critical: {current_memory / 1024 / 1024:.2f}MB "
                              f"> {self._critical_threshold * 100}% threshold")
                self._emergency_release()
            
            # 如果超过警告阈值，尝试释放低优先级模型
            elif current_memory > self._max_memory * self._warning_threshold:
                logger.warning(f"Memory warning: {current_memory / 1024 / 1024:.2f}MB "
                              f"> {self._warning_threshold * 100}% threshold")
                self._release_low_priority_models()
            
            # 加载新模型
            logger.info(f"Loading model: {cache_key}, priority={priority.value}")
            
            try:
                model = loader_func(model_name)
                
                # 估算模型内存占用
                memory_estimate = self._estimate_model_size(model)
                
                # 存储模型和信息
                self._models[cache_key] = model
                self._model_info[cache_key] = ModelInfo(
                    name=model_name,
                    model_type=model_type,
                    loaded_at=time.time(),
                    last_used=time.time(),
                    memory_estimate=memory_estimate,
                    priority=priority,
                    usage_count=1
                )
                
                self._total_loads += 1
                
                logger.info(f"Model loaded: {cache_key}, "
                           f"estimated_memory={memory_estimate / 1024 / 1024:.2f}MB")
                
                return model
            
            except Exception as e:
                logger.error(f"Failed to load model {cache_key}: {str(e)}")
                raise
    
    def release_model(self, model_type: str, model_name: str) -> bool:
        """
        释放指定模型
        
        Args:
            model_type: 模型类型
            model_name: 模型名称
        
        Returns:
            是否成功释放
        """
        cache_key = f"{model_type}_{model_name}"
        
        with self._lock:
            if cache_key not in self._models:
                logger.warning(f"Model not found: {cache_key}")
                return False
            
            # 检查优先级
            priority = self._model_info[cache_key].priority
            
            if priority == ModelPriority.HIGH:
                logger.warning(f"Cannot release high priority model: {cache_key}")
                return False
            
            # 释放模型
            logger.info(f"Releasing model: {cache_key}")
            
            try:
                del self._models[cache_key]
                del self._model_info[cache_key]
                
                # 清理 GPU 内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 清理 Metal 内存 (MacOS)
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                
                self._total_releases += 1
                
                logger.info(f"Model released: {cache_key}")
                
                return True
            
            except Exception as e:
                logger.error(f"Failed to release model {cache_key}: {str(e)}")
                return False
    
    def get_model(self, model_type: str, model_name: str) -> Optional[Any]:
        """
        获取已加载的模型
        
        Args:
            model_type: 模型类型
            model_name: 模型名称
        
        Returns:
            模型实例，如果不存在则返回 None
        """
        cache_key = f"{model_type}_{model_name}"
        
        with self._lock:
            if cache_key in self._models:
                # 更新使用信息
                self._model_info[cache_key].last_used = time.time()
                self._model_info[cache_key].usage_count += 1
                
                return self._models[cache_key]
            
            return None
    
    def has_model(self, model_type: str, model_name: str) -> bool:
        """
        检查模型是否已加载
        
        Args:
            model_type: 模型类型
            model_name: 模型名称
        
        Returns:
            是否已加载
        """
        cache_key = f"{model_type}_{model_name}"
        
        with self._lock:
            return cache_key in self._models
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        获取内存统计信息
        
        Returns:
            内存统计字典
        """
        current_memory = psutil.Process().memory_info().rss
        
        with self._lock:
            loaded_models = len(self._models)
            total_model_memory = sum(
                info.memory_estimate for info in self._model_info.values()
            )
            
            models_info = [
                {
                    'name': info.name,
                    'type': info.model_type,
                    'loaded_at': info.loaded_at,
                    'last_used': info.last_used,
                    'memory_mb': info.memory_estimate / 1024 / 1024,
                    'priority': info.priority.value,
                    'usage_count': info.usage_count
                }
                for info in self._model_info.values()
            ]
        
        return {
            'current_memory_mb': current_memory / 1024 / 1024,
            'max_memory_mb': self._max_memory / 1024 / 1024,
            'peak_memory_mb': self._peak_memory / 1024 / 1024,
            'memory_usage_percent': round(current_memory / self._max_memory * 100, 2),
            'loaded_models': loaded_models,
            'total_model_memory_mb': total_model_memory / 1024 / 1024,
            'models': models_info,
            'total_loads': self._total_loads,
            'total_releases': self._total_releases
        }
    
    def start_auto_release(self):
        """
        启动自动释放线程
        """
        if self._auto_release_enabled and not self._running:
            self._running = True
            self._release_thread = threading.Thread(
                target=self._auto_release_loop,
                daemon=True
            )
            self._release_thread.start()
            
            logger.info("Auto-release thread started")
    
    def stop_auto_release(self):
        """
        停止自动释放线程
        """
        if self._running:
            self._running = False
            
            if self._release_thread:
                self._release_thread.join(timeout=5)
            
            logger.info("Auto-release thread stopped")
    
    def _auto_release_loop(self):
        """
        自动释放循环
        """
        while self._running:
            try:
                time.sleep(self._release_interval)
                
                # 检查内存使用
                current_memory = psutil.Process().memory_info().rss
                
                if current_memory > self._max_memory * self._warning_threshold:
                    logger.info("Auto-release triggered by memory threshold")
                    self._release_oldest_unused_models()
                
            except Exception as e:
                logger.error(f"Error in auto-release loop: {str(e)}")
    
    def _release_oldest_unused_models(self, max_age: int = 600):
        """
        释放最久未使用的模型
        
        Args:
            max_age: 最大未使用时间 (秒)
        """
        current_time = time.time()
        
        with self._lock:
            # 找出最久未使用的低优先级模型
            candidates = []
            
            for cache_key, info in self._model_info.items():
                if info.priority != ModelPriority.HIGH:
                    age = current_time - info.last_used
                    
                    if age > max_age:
                        candidates.append((cache_key, age))
            
            # 按年龄排序
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # 释放最久的模型
            for cache_key, age in candidates:
                logger.info(f"Auto-releasing unused model: {cache_key}, "
                           f"age={age:.2f}s")
                
                try:
                    del self._models[cache_key]
                    del self._model_info[cache_key]
                    
                    # 清理 GPU 内存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    self._total_releases += 1
                    
                except Exception as e:
                    logger.error(f"Failed to auto-release model {cache_key}: {str(e)}")
    
    def _release_low_priority_models(self):
        """
        释放所有低优先级模型
        """
        with self._lock:
            low_priority_keys = [
                cache_key for cache_key, info in self._model_info.items()
                if info.priority == ModelPriority.LOW
            ]
            
            for cache_key in low_priority_keys:
                logger.info(f"Releasing low priority model: {cache_key}")
                
                try:
                    del self._models[cache_key]
                    del self._model_info[cache_key]
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    self._total_releases += 1
                    
                except Exception as e:
                    logger.error(f"Failed to release model {cache_key}: {str(e)}")
    
    def _emergency_release(self):
        """
        紧急释放（释放所有非高优先级模型）
        """
        logger.warning("Emergency release triggered")
        
        with self._lock:
            # 释放所有非高优先级模型
            release_keys = [
                cache_key for cache_key, info in self._model_info.items()
                if info.priority != ModelPriority.HIGH
            ]
            
            for cache_key in release_keys:
                logger.warning(f"Emergency releasing model: {cache_key}")
                
                try:
                    del self._models[cache_key]
                    del self._model_info[cache_key]
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    self._total_releases += 1
                    
                except Exception as e:
                    logger.error(f"Failed to emergency release model {cache_key}: {str(e)}")
    
    def _estimate_model_size(self, model: Any) -> int:
        """
        估算模型内存占用
        
        Args:
            model: 模型实例
        
        Returns:
            估算的内存占用 (bytes)
        """
        try:
            # PyTorch 模型
            if hasattr(model, 'parameters'):
                param_size = sum(
                    p.numel() * p.element_size()
                    for p in model.parameters()
                )
                
                # 加上缓冲区大小
                buffer_size = sum(
                    b.numel() * b.element_size()
                    for b in model.buffers()
                )
                
                # 总大小（乘以2，考虑梯度等）
                total_size = (param_size + buffer_size) * 2
                
                return total_size
            
            # Whisper 模型
            if hasattr(model, 'dims'):
                # Whisper 模型大小估算
                model_sizes = {
                    'tiny': 39 * 1024 * 1024,
                    'base': 74 * 1024 * 1024,
                    'small': 244 * 1024 * 1024,
                    'medium': 769 * 1024 * 1024,
                    'large': 1550 * 1024 * 1024,
                    'large-v3': 2900 * 1024 * 1024
                }
                
                # 根据模型维度估算
                if hasattr(model, 'name'):
                    return model_sizes.get(model.name, 500 * 1024 * 1024)
                
                return 500 * 1024 * 1024
            
            # Demucs 模型
            if hasattr(model, 'sources'):
                # Demucs 模型大小估算
                return 1000 * 1024 * 1024  # ~1GB
            
            # 默认估算
            return 100 * 1024 * 1024  # 100MB
        
        except Exception as e:
            logger.error(f"Error estimating model size: {str(e)}")
            return 100 * 1024 * 1024  # 默认 100MB
    
    def clear_all(self):
        """
        清空所有模型
        """
        with self._lock:
            logger.info("Clearing all models")
            
            self._models.clear()
            self._model_info.clear()
            
            # 清理 GPU 内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 清理 Metal 内存
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            logger.info("All models cleared")

# 全局模型管理器实例
model_manager = ModelManager(
    max_memory_gb=4.0,
    warning_threshold=0.8,
    critical_threshold=0.9,
    auto_release_enabled=True,
    release_interval=300
)

# 启动自动释放
model_manager.start_auto_release()