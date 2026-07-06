# local-media-processor Phase 1 & Phase 3 实施指南

**实施时间**: 2026-06-20
**项目路径**: `/Users/donghuiwang/HomeServer/local-media-processor`

---

## 📋 已完成的工作

### ✅ Phase 1: 任务队列系统

1. **创建任务队列系统** ✅
   - 文件: [backend/task_queue.py](file:///Users/donghuiwang/HomeServer/local-media-processor/backend/task_queue.py)
   - 功能: Celery 任务定义、任务管理函数
   - 包含: 8 个异步任务（音频、图像、视频处理）

2. **更新 Docker 配置** ✅
   - 文件: [docker-compose.yml](file:///Users/donghuiwang/HomeServer/local-media-processor/docker-compose.yml)
   - 新增服务:
     - Redis 服务（任务队列和结果存储）
     - Celery Worker 服务（任务处理）
   - 配置: 健康检查、依赖关系、网络配置

3. **更新依赖** ✅
   - 文件: [requirements.txt](file:///Users/donghuiwang/HomeServer/local-media-processor/requirements.txt)
   - 新增依赖:
     - celery==5.3.4
     - redis==5.0.1
     - kombu==5.3.3
     - psutil==5.9.7
     - prometheus-client==0.19.0

---

### ✅ Phase 3: 模型管理器

1. **创建模型管理器** ✅
   - 文件: [backend/model_manager.py](file:///Users/donghuiwang/HomeServer/local-media-processor/backend/model_manager.py)
   - 功能:
     - 统一模型加载和缓存
     - 内存监控和管理
     - 自动释放机制
     - 模型生命周期管理
     - 优先级管理

2. **创建监控指标端点** ✅
   - 文件: [backend/metrics.py](file:///Users/donghuiwang/HomeServer/local-media-processor/backend/metrics.py)
   - 功能:
     - Prometheus 指标端点
     - 系统状态统计
     - 健康检查端点
     - 实时监控数据

---

## 🚀 下一步实施步骤

### Step 1: 修改 API 端点使用任务队列

**目标**: 将现有的同步 API 端点改为异步任务提交

**需要修改的文件**: `backend/main.py`

**修改示例**:

```python
# 原来的同步端点
@app.post("/api/audio/process")
async def process_audio_endpoint(...):
    result = process_audio(input_path, output_path, ...)
    return {"success": result["success"], ...}

# 改为异步端点
@app.post("/api/audio/process")
async def process_audio_endpoint(...):
    from .task_queue import process_audio_task
    
    # 提交任务到队列
    task = process_audio_task.delay(
        input_path,
        output_path,
        {
            "silence_threshold": silence_threshold,
            "min_silence_duration": min_silence_duration,
            "max_volume": max_volume
        }
    )
    
    return {
        "task_id": task.id,
        "status": "PENDING",
        "message": "任务已提交，请使用任务ID查询进度"
    }
```

**需要修改的端点**:
- `/api/audio/process` → `process_audio_task`
- `/api/audio/transcribe` → `transcribe_audio_task`
- `/api/audio/separate` → `separate_audio_task`
- `/api/image/process` → `process_image_task`
- `/api/image/upscale` → `upscale_image_task`
- `/api/image/enhance-face` → `enhance_face_task`
- `/api/video/process` → `process_video_task`

---

### Step 2: 创建任务状态查询端点

**目标**: 提供任务进度查询功能

**在 `backend/main.py` 中添加**:

```python
from .task_queue import get_task_status, cancel_task, get_active_tasks, get_queue_stats
from .metrics import metrics_router

# 注册监控路由
app.include_router(metrics_router)

@app.get("/api/tasks/{task_id}")
async def get_task_status_endpoint(task_id: str):
    """查询任务状态"""
    status = get_task_status(task_id)
    return status

@app.delete("/api/tasks/{task_id}")
async def cancel_task_endpoint(task_id: str):
    """取消任务"""
    success = cancel_task(task_id)
    return {
        "success": success,
        "message": "任务已取消" if success else "取消失败"
    }

@app.get("/api/tasks/active")
async def get_active_tasks_endpoint():
    """获取活跃任务列表"""
    tasks = get_active_tasks()
    return {"tasks": tasks}

@app.get("/api/tasks/stats")
async def get_queue_stats_endpoint():
    """获取队列统计"""
    stats = get_queue_stats()
    return stats
```

---

### Step 3: 修改处理器使用模型管理器

**目标**: 让处理器通过模型管理器加载模型

**需要修改的文件**:
- `backend/whisper_processor.py`
- `backend/demucs_processor.py`
- `backend/esrgan_processor.py`

**修改示例 (whisper_processor.py)**:

```python
# 原来的模型缓存
_model_cache = {}

def get_model(model_name="large-v3"):
    if model_name not in _model_cache:
        _model_cache[model_name] = whisper.load_model(model_name)
    return _model_cache[model_name]

# 改为使用模型管理器
from .model_manager import model_manager, ModelPriority

def get_model(model_name="large-v3"):
    """通过模型管理器加载模型"""
    return model_manager.load_model(
        'whisper',
        model_name,
        lambda name: whisper.load_model(name),
        priority=ModelPriority.MEDIUM
    )
```

**修改示例 (demucs_processor.py)**:

```python
from .model_manager import model_manager, ModelPriority

def get_demucs_model(model_name="htdemucs"):
    """通过模型管理器加载模型"""
    return model_manager.load_model(
        'demucs',
        model_name,
        lambda name: get_model(name),
        priority=ModelPriority.LOW  # Demucs 模型优先级较低，容易释放
    )
```

**修改示例 (esrgan_processor.py)**:

```python
from .model_manager import model_manager, ModelPriority

def get_upsampler(model_name="RealESRGAN_x4plus", scale=4):
    """通过模型管理器加载放大器"""
    cache_key = f"{model_name}_{scale}"
    
    # 尝试从模型管理器获取
    upsampler = model_manager.get_model('esrgan', cache_key)
    
    if upsampler is None:
        # 创建新的放大器
        upsampler = _create_upsampler(model_name, scale)
        
        # 存储到模型管理器
        model_manager._models[f"esrgan_{cache_key}"] = upsampler
        model_manager._model_info[f"esrgan_{cache_key}"] = ModelInfo(
            name=cache_key,
            model_type='esrgan',
            loaded_at=time.time(),
            last_used=time.time(),
            memory_estimate=_estimate_upsampler_size(upsampler),
            priority=ModelPriority.MEDIUM
        )
    
    return upsampler
```

---

### Step 4: 启动和测试

**启动服务**:

```bash
# 进入项目目录
cd /Users/donghuiwang/HomeServer/local-media-processor

# 构建并启动所有服务
docker-compose up -d --build

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f celery-worker
docker-compose logs -f redis
```

**测试任务队列**:

```bash
# 测试 API 端点（提交任务）
curl -X POST http://localhost:8000/api/audio/process \
  -F "file=@test.mp3" \
  -F "silence_threshold=-40.0" \
  -F "min_silence_duration=0.5" \
  -F "max_volume=true"

# 返回示例:
# {
#   "task_id": "abc123",
#   "status": "PENDING",
#   "message": "任务已提交"
# }

# 查询任务状态
curl http://localhost:8000/api/tasks/abc123

# 返回示例:
# {
#   "task_id": "abc123",
#   "status": "PROCESSING",
#   "progress": 50,
#   "stage": "processing",
#   "message": "正在处理音频"
# }
```

**测试监控指标**:

```bash
# Prometheus 指标
curl http://localhost:8000/metrics

# 系统状态
curl http://localhost:8000/api/system/stats

# 健康检查
curl http://localhost:8000/api/system/health
```

---

## 📊 预期收益

### 性能提升

| 指标 | 当前 | 优化后 | 提升 |
|------|------|--------|------|
| API 响应时间 | 30-300秒 | <1秒 | **95%+** |
| 并发处理能力 | 1个任务 | 多个任务 | **100%+** |
| 任务进度追踪 | 无 | 实时 | **100%** |

### 内存优化

| 指标 | 当前 | 优化后 | 提升 |
|------|------|--------|------|
| 内存占用 | 5-8GB | 3-4GB | **30-40%** |
| 模型加载速度 | 慢 | 快（缓存） | **50%** |
| 内存溢出风险 | 高 | 低（自动释放） | **80%** |

---

## 🔧 配置说明

### Redis 配置

```yaml
# docker-compose.yml
redis:
  image: redis:7-alpine
  command: redis-server --appendonly yes --maxmemory 2gb --maxmemory-policy allkeys-lru
```

**参数说明**:
- `--appendonly yes`: 数据持久化
- `--maxmemory 2gb`: 最大内存 2GB
- `--maxmemory-policy allkeys-lru`: 内存满时使用 LRU 算法删除键

---

### Celery Worker 配置

```yaml
# docker-compose.yml
celery-worker:
  command: celery -A backend.task_queue worker --loglevel=info --concurrency=2 --max-tasks-per-child=50
```

**参数说明**:
- `--concurrency=2`: 同时处理 2 个任务
- `--max-tasks-per-child=50`: 每个 worker 处理 50 个任务后重启（防止内存泄漏）
- `--loglevel=info`: 日志级别

---

### 模型管理器配置

```python
# backend/model_manager.py
model_manager = ModelManager(
    max_memory_gb=4.0,        # 最大内存 4GB
    warning_threshold=0.8,    # 警告阈值 80%
    critical_threshold=0.9,   # 临界阈值 90%
    auto_release_enabled=True, # 启用自动释放
    release_interval=300      # 检查间隔 5分钟
)
```

---

## 🐛 常见问题

### Q1: Redis 连接失败

**症状**: `celery-worker` 无法启动，报错 "Connection refused"

**解决方案**:
```bash
# 检查 Redis 是否运行
docker-compose ps redis

# 重启 Redis
docker-compose restart redis

# 检查 Redis 日志
docker-compose logs redis
```

---

### Q2: 任务一直处于 PENDING 状态

**症状**: 提交任务后，状态一直是 PENDING

**解决方案**:
```bash
# 检查 Celery Worker 是否运行
docker-compose ps celery-worker

# 检查 Worker 日志
docker-compose logs celery-worker

# 重启 Worker
docker-compose restart celery-worker
```

---

### Q3: 内存占用过高

**症状**: 内存使用超过 80%

**解决方案**:
```bash
# 查看内存统计
curl http://localhost:8000/api/system/stats

# 查看模型状态
curl http://localhost:8000/api/system/stats | jq '.models'

# 手动释放模型（需要添加端点）
# 或等待自动释放（5分钟后）
```

---

## 📈 监控建议

### Prometheus + Grafana

1. **配置 Prometheus**:
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'media-processor'
    static_configs:
      - targets: ['localhost:8000']
```

2. **导入 Grafana Dashboard**:
- 使用 Prometheus 数据源
- 导入标准 Dashboard（ID: 1860）
- 自定义面板显示内存、任务、模型指标

---

### 告警规则

```yaml
# alert_rules.yml
groups:
  - name: media-processor-alerts
    rules:
      - alert: HighMemoryUsage
        expr: process_memory_bytes > 3.2e9  # 3.2GB
        for: 5m
        annotations:
          summary: "内存使用过高"

      - alert: TaskQueueFull
        expr: task_queue_size > 10
        for: 5m
        annotations:
          summary: "任务队列积压"

      - alert: CeleryWorkerDown
        expr: active_tasks_count == 0
        for: 10m
        annotations:
          summary: "Celery Worker 可能已停止"
```

---

## ✅ 实施完成确认

- ✅ **Phase 1**: 任务队列系统已创建
- ✅ **Phase 3**: 模型管理器已创建
- ✅ **Docker 配置**: Redis 和 Celery Worker 已配置
- ✅ **依赖更新**: Celery、Redis、监控依赖已添加
- ✅ **监控指标**: Prometheus 指标端点已创建

**下一步**: 按照实施步骤修改 API 端点和处理器，完成完整集成。

---

**实施指南创建时间**: 2026-06-20
**项目**: local-media-processor
**状态**: Phase 1 & Phase 3 核心组件已创建，待集成