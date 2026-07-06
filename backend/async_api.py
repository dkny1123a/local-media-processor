"""
异步 API 端点 - 使用任务队列处理媒体文件
提供异步处理、任务状态查询、进度追踪等功能
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import os
import uuid
import time
from typing import Optional

async_router = APIRouter(prefix="/async", tags=["Async Processing"])

from .task_queue import (
    process_audio_task,
    process_media_task,
    get_task_status,
    cancel_task,
    get_active_tasks,
    get_queue_stats
)

from .metrics import (
    metrics_router,
    record_processing_request,
    record_processing_time
)

from .utils import validate_file_type, generate_unique_id, cleanup_temp_files

def detect_file_type(filename):
    ext = os.path.splitext(filename)[1].lower()
    audio_exts = ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.amr', '.3gp']
    video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    if ext in audio_exts:
        return 'audio'
    elif ext in video_exts:
        return 'video'
    else:
        return None

@async_router.post("/media/process")
async def async_process_media(
    file: UploadFile = File(...),
    silence_threshold: float = Form(-40.0),
    min_silence_duration: float = Form(0.5),
    max_volume: bool = Form(True)
):
    """
    异步媒体处理端点 - 统一入口
    根据文件类型自动路由到音频或视频处理
    
    Args:
        file: 上传的媒体文件（音频或视频）
        silence_threshold: 静音阈值 (dB)
        min_silence_duration: 最小静音时长 (秒)
        max_volume: 是否最大化音量
    
    Returns:
        任务ID和状态信息
    """
    file_type = detect_file_type(file.filename)
    
    if not file_type:
        raise HTTPException(status_code=400, detail="不支持的文件格式，请上传音频或视频文件")
    
    task_id = generate_unique_id()
    
    upload_dir = os.environ.get('UPLOAD_DIR', '/app/uploads')
    input_path = os.path.join(upload_dir, f"{task_id}_{file.filename}")
    
    with open(input_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    output_dir = os.environ.get('OUTPUT_DIR', '/app/output')
    base_name = os.path.splitext(file.filename)[0]
    
    if file_type == 'audio':
        ext = os.path.splitext(file.filename)[1]
        output_filename = f"{base_name}_processed{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        task = process_media_task.delay(
            input_path,
            output_path,
            {
                "file_type": "audio",
                "silence_threshold": silence_threshold,
                "min_silence_duration": min_silence_duration,
                "max_volume": max_volume
            }
        )
        
        record_processing_request('audio', 'process', 'submitted')
        
        return {
            "task_id": task.id,
            "status": "PENDING",
            "message": "音频处理任务已提交",
            "input_file": file.filename,
            "file_type": "audio",
            "output_file": output_filename,
            "options": {
                "silence_threshold": silence_threshold,
                "min_silence_duration": min_silence_duration,
                "max_volume": max_volume
            }
        }
    else:
        audio_output_path = os.path.join(output_dir, f"{base_name}_audio.mp3")
        processed_audio_path = os.path.join(output_dir, f"{base_name}_processed.mp3")
        
        task = process_media_task.delay(
            input_path,
            processed_audio_path,
            {
                "file_type": "video",
                "audio_output_path": audio_output_path,
                "silence_threshold": silence_threshold,
                "min_silence_duration": min_silence_duration,
                "max_volume": max_volume
            }
        )
        
        record_processing_request('video', 'process', 'submitted')
        
        return {
            "task_id": task.id,
            "status": "PENDING",
            "message": "视频处理任务已提交",
            "input_file": file.filename,
            "file_type": "video",
            "output_files": {
                "audio": os.path.basename(audio_output_path),
                "processed_audio": os.path.basename(processed_audio_path)
            },
            "options": {
                "silence_threshold": silence_threshold,
                "min_silence_duration": min_silence_duration,
                "max_volume": max_volume
            }
        }

@async_router.get("/tasks/{task_id}")
async def get_task_status_endpoint(task_id: str):
    """
    查询任务状态
    返回任务的详细状态和进度信息
    
    Args:
        task_id: 任务ID
    
    Returns:
        任务状态信息
    """
    status = get_task_status(task_id)
    
    if status['status'] == 'UNKNOWN':
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return status

@async_router.delete("/tasks/{task_id}")
async def cancel_task_endpoint(task_id: str):
    """
    取消任务
    终止正在执行或等待的任务
    
    Args:
        task_id: 任务ID
    
    Returns:
        取消结果
    """
    success = cancel_task(task_id)
    
    if not success:
        raise HTTPException(status_code=400, detail="无法取消任务")
    
    return {
        "success": True,
        "task_id": task_id,
        "message": "任务已取消"
    }

@async_router.get("/tasks/active")
async def get_active_tasks_endpoint():
    """
    获取活跃任务列表
    返回当前正在执行的任务
    
    Returns:
        活跃任务列表
    """
    tasks = get_active_tasks()
    
    return {
        "count": len(tasks),
        "tasks": tasks
    }

@async_router.get("/tasks/stats")
async def get_queue_stats_endpoint():
    """
    获取队列统计信息
    返回任务队列的统计数据
    
    Returns:
        队列统计信息
    """
    stats = get_queue_stats()
    
    return {
        "timestamp": time.time(),
        "stats": stats
    }

@async_router.post("/batch/audio/process")
async def batch_process_audio(
    files: list[UploadFile] = File(...),
    silence_threshold: float = Form(-40.0),
    min_silence_duration: float = Form(0.5),
    max_volume: bool = Form(True)
):
    """
    批量音频处理端点
    同时处理多个音频文件
    
    Args:
        files: 上传的音频文件列表
        silence_threshold: 静音阈值 (dB)
        min_silence_duration: 最小静音时长 (秒)
        max_volume: 是否最大化音量
    
    Returns:
        任务ID列表
    """
    task_ids = []
    
    for file in files:
        if not validate_file_type(file.filename, ['audio']):
            continue
        
        task_id = generate_unique_id()
        
        upload_dir = os.environ.get('UPLOAD_DIR', '/app/uploads')
        input_path = os.path.join(upload_dir, f"{task_id}_{file.filename}")
        
        with open(input_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        output_dir = os.environ.get('OUTPUT_DIR', '/app/output')
        output_filename = f"processed_{task_id}_{file.filename}"
        output_path = os.path.join(output_dir, output_filename)
        
        task = process_audio_task.delay(
            input_path,
            output_path,
            {
                "silence_threshold": silence_threshold,
                "min_silence_duration": min_silence_duration,
                "max_volume": max_volume
            }
        )
        
        task_ids.append({
            "task_id": task.id,
            "input_file": file.filename,
            "output_file": output_filename
        })
    
    record_processing_request('audio', 'batch_process', 'submitted')
    
    return {
        "count": len(task_ids),
        "tasks": task_ids,
        "message": f"{len(task_ids)} 个音频处理任务已提交"
    }