"""
监控指标端点 - 提供 Prometheus 指标和系统状态监控
支持性能监控、任务统计、内存监控
"""
from fastapi import APIRouter, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import psutil
import time

# 创建 API Router
metrics_router = APIRouter()

# ==================== Prometheus 指标定义 ====================

# 处理请求计数
PROCESSING_REQUESTS = Counter(
    'media_processing_requests_total',
    'Total processing requests',
    ['media_type', 'operation', 'status']
)

# 处理时间
PROCESSING_TIME = Histogram(
    'media_processing_duration_seconds',
    'Processing duration in seconds',
    ['media_type', 'operation'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0]
)

# 模型加载计数
MODEL_LOADED = Counter(
    'ai_models_loaded_total',
    'Total AI models loaded',
    ['model_type', 'model_name']
)

# 模型释放计数
MODEL_RELEASED = Counter(
    'ai_models_released_total',
    'Total AI models released',
    ['model_type', 'model_name']
)

# 当前加载的模型数量
CURRENT_MODELS = Gauge(
    'ai_models_current',
    'Number of AI models currently loaded',
    ['model_type']
)

# 内存使用
MEMORY_USAGE = Gauge(
    'process_memory_bytes',
    'Current process memory usage in bytes'
)

# 内存峰值
MEMORY_PEAK = Gauge(
    'process_memory_peak_bytes',
    'Peak process memory usage in bytes'
)

# 任务队列大小
QUEUE_SIZE = Gauge(
    'task_queue_size',
    'Current task queue size'
)

# 活跃任务数
ACTIVE_TASKS = Gauge(
    'active_tasks_count',
    'Number of active tasks'
)

# CPU 使用率
CPU_USAGE = Gauge(
    'process_cpu_percent',
    'Current process CPU usage percentage'
)

# ==================== API 端点 ====================

@metrics_router.get("/metrics")
async def prometheus_metrics():
    """
    Prometheus 指标端点
    返回 Prometheus 格式的监控指标
    """
    # 更新实时指标
    process = psutil.Process()
    
    # 内存指标
    memory_info = process.memory_info()
    MEMORY_USAGE.set(memory_info.rss)
    
    # CPU 指标
    CPU_USAGE.set(process.cpu_percent())
    
    # 任务队列指标（如果可用）
    try:
        from .task_queue import get_queue_stats
        queue_stats = get_queue_stats()
        QUEUE_SIZE.set(queue_stats.get('reserved_tasks', 0))
        ACTIVE_TASKS.set(queue_stats.get('active_tasks', 0))
    except Exception:
        pass
    
    # 模型管理器指标（如果可用）
    try:
        from .model_manager import model_manager
        memory_stats = model_manager.get_memory_stats()
        MEMORY_PEAK.set(memory_stats.get('peak_memory_mb', 0) * 1024 * 1024)
        
        # 更新当前模型数量
        for model_type in ['whisper', 'demucs']:
            count = sum(
                1 for info in memory_stats.get('models', [])
                if info.get('type') == model_type
            )
            CURRENT_MODELS.labels(model_type=model_type).set(count)
    except Exception:
        pass
    
    # 返回 Prometheus 格式的指标
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )

@metrics_router.get("/api/system/stats")
async def system_stats():
    """
    系统状态统计
    返回详细的系统状态信息
    """
    process = psutil.Process()
    
    # 系统信息
    system_info = {
        "cpu_count": psutil.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_total": psutil.virtual_memory().total,
        "memory_available": psutil.virtual_memory().available,
        "memory_percent": psutil.virtual_memory().percent,
        "disk_total": psutil.disk_usage('/').total,
        "disk_used": psutil.disk_usage('/').used,
        "disk_percent": psutil.disk_usage('/').percent
    }
    
    # 进程信息
    process_info = {
        "pid": process.pid,
        "cpu_percent": process.cpu_percent(interval=1),
        "memory_rss": process.memory_info().rss,
        "memory_vms": process.memory_info().vms,
        "memory_percent": process.memory_percent(),
        "num_threads": process.num_threads(),
        "num_fds": process.num_fds() if hasattr(process, 'num_fds') else 0,
        "create_time": process.create_time(),
        "running_time": time.time() - process.create_time()
    }
    
    # 任务队列信息
    try:
        from .task_queue import get_queue_stats, get_active_tasks
        queue_stats = get_queue_stats()
        active_tasks = get_active_tasks()
        
        task_info = {
            "active_tasks": queue_stats.get('active_tasks', 0),
            "reserved_tasks": queue_stats.get('reserved_tasks', 0),
            "scheduled_tasks": queue_stats.get('scheduled_tasks', 0),
            "active_tasks_list": active_tasks[:10]  # 只显示前10个
        }
    except Exception as e:
        task_info = {
            "error": str(e),
            "active_tasks": 0,
            "reserved_tasks": 0,
            "scheduled_tasks": 0
        }
    
    # 模型管理器信息
    try:
        from .model_manager import model_manager
        memory_stats = model_manager.get_memory_stats()
        
        model_info = {
            "loaded_models": memory_stats.get('loaded_models', 0),
            "total_loads": memory_stats.get('total_loads', 0),
            "total_releases": memory_stats.get('total_releases', 0),
            "current_memory_mb": memory_stats.get('current_memory_mb', 0),
            "peak_memory_mb": memory_stats.get('peak_memory_mb', 0),
            "memory_usage_percent": memory_stats.get('memory_usage_percent', 0),
            "models": memory_stats.get('models', [])
        }
    except Exception as e:
        model_info = {
            "error": str(e),
            "loaded_models": 0
        }
    
    return {
        "timestamp": time.time(),
        "system": system_info,
        "process": process_info,
        "tasks": task_info,
        "models": model_info
    }

@metrics_router.get("/api/system/health")
async def health_check():
    """
    健康检查端点
    返回服务健康状态
    """
    # 检查 Redis 连接
    redis_status = "unknown"
    try:
        import redis
        import os
        redis_host = os.environ.get('REDIS_HOST', 'redis')
        redis_port = int(os.environ.get('REDIS_PORT', 6379))
        redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            socket_connect_timeout=2
        )
        redis_client.ping()
        redis_status = "healthy"
    except Exception as e:
        redis_status = f"unhealthy: {str(e)}"
    
    # 检查 Celery Worker
    celery_status = "unknown"
    try:
        from .task_queue import get_queue_stats
        queue_stats = get_queue_stats()
        celery_status = "healthy" if queue_stats else "degraded"
    except Exception as e:
        celery_status = f"unhealthy: {str(e)}"
    
    # 检查内存使用
    process = psutil.Process()
    memory_percent = process.memory_percent()
    
    memory_status = "healthy" if memory_percent < 80 else (
        "warning" if memory_percent < 90 else "critical"
    )
    
    # 整体健康状态
    overall_status = "healthy"
    if redis_status != "healthy" or celery_status != "healthy":
        overall_status = "degraded"
    if memory_status == "critical":
        overall_status = "critical"
    
    return {
        "status": overall_status,
        "timestamp": time.time(),
        "components": {
            "redis": {
                "status": redis_status
            },
            "celery": {
                "status": celery_status
            },
            "memory": {
                "status": memory_status,
                "percent": round(memory_percent, 2)
            }
        }
    }

# ==================== 辅助函数 ====================

def record_processing_request(media_type: str, operation: str, status: str):
    """
    记录处理请求
    
    Args:
        media_type: 媒体类型 (audio, image, video)
        operation: 操作类型
        status: 状态 (success, failed)
    """
    PROCESSING_REQUESTS.labels(
        media_type=media_type,
        operation=operation,
        status=status
    ).inc()

def record_processing_time(media_type: str, operation: str, duration: float):
    """
    记录处理时间
    
    Args:
        media_type: 媒体类型
        operation: 操作类型
        duration: 处理时间 (秒)
    """
    PROCESSING_TIME.labels(
        media_type=media_type,
        operation=operation
    ).observe(duration)

def record_model_loaded(model_type: str, model_name: str):
    """
    记录模型加载
    
    Args:
        model_type: 模型类型
        model_name: 模型名称
    """
    MODEL_LOADED.labels(
        model_type=model_type,
        model_name=model_name
    ).inc()

def record_model_released(model_type: str, model_name: str):
    """
    记录模型释放
    
    Args:
        model_type: 模型类型
        model_name: 模型名称
    """
    MODEL_RELEASED.labels(
        model_type=model_type,
        model_name=model_name
    ).inc()