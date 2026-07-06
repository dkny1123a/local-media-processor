"""
任务队列系统 - 使用 Celery 实现异步处理
支持音频和视频的异步处理和进度跟踪
"""
from celery import Celery
import os
import time
from typing import Dict, Any

celery_app = Celery(
    'media_processor',
    broker=os.environ.get('CELERY_BROKER', 'redis://redis:6379/0'),
    backend=os.environ.get('CELERY_BACKEND', 'redis://redis:6379/1')
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,
    task_soft_time_limit=3300,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)

@celery_app.task(bind=True, name='process_audio')
def process_audio_task(
    self,
    input_path: str,
    output_path: str,
    options: Dict[str, Any]
) -> Dict[str, Any]:
    """
    异步音频处理任务
    
    Args:
        self: Celery task instance
        input_path: 输入音频文件路径
        output_path: 输出音频文件路径
        options: 处理选项
            - silence_threshold: 静音阈值 (dB)
            - min_silence_duration: 最小静音时长 (秒)
            - max_volume: 是否最大化音量
    
    Returns:
        处理结果字典
    """
    from .audio_processor import process_audio
    from .model_manager import model_manager
    
    try:
        self.update_state(
            state='PROCESSING',
            meta={
                'progress': 0,
                'stage': 'loading',
                'message': '正在加载音频文件'
            }
        )
        
        start_time = time.time()
        
        self.update_state(
            state='PROCESSING',
            meta={
                'progress': 20,
                'stage': 'processing',
                'message': '正在处理音频'
            }
        )
        
        result = process_audio(
            input_path,
            output_path,
            silence_threshold=options.get('silence_threshold', -40.0),
            min_silence_duration=options.get('min_silence_duration', 0.5),
            max_volume=options.get('max_volume', True)
        )
        
        processing_time = time.time() - start_time
        
        self.update_state(
            state='PROCESSING',
            meta={
                'progress': 100,
                'stage': 'completed',
                'message': '音频处理完成',
                'processing_time': round(processing_time, 2)
            }
        )
        
        if result['success']:
            result['stats']['processing_time'] = round(processing_time, 2)
        
        return result
    
    except Exception as e:
        self.update_state(
            state='FAILED',
            meta={
                'error': str(e),
                'error_type': type(e).__name__,
                'stage': 'failed'
            }
        )
        
        return {
            'success': False,
            'message': f'音频处理失败: {str(e)}',
            'error_type': type(e).__name__
        }



@celery_app.task(bind=True, name='process_media')
def process_media_task(
    self,
    input_path: str,
    output_path: str,
    options: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    统一媒体处理任务 - 根据文件类型自动路由
    
    Args:
        self: Celery task instance
        input_path: 输入媒体文件路径
        output_path: 输出文件路径
        options: 处理选项
            - file_type: 文件类型 (audio/video)
            - audio_output_path: 视频提取音频时的输出路径
            - silence_threshold: 静音阈值 (dB)
            - min_silence_duration: 最小静音时长 (秒)
            - max_volume: 是否最大化音量
    
    Returns:
        处理结果字典
    """
    try:
        options = options or {}
        file_type = options.get('file_type', 'audio')
        
        if file_type == 'audio':
            return process_audio_task(
                self,
                input_path,
                output_path,
                {
                    'silence_threshold': options.get('silence_threshold', -40.0),
                    'min_silence_duration': options.get('min_silence_duration', 0.5),
                    'max_volume': options.get('max_volume', True)
                }
            )
        else:
            from .video_processor import process_video
            
            audio_output_path = options.get('audio_output_path', output_path)
            
            self.update_state(
                state='PROCESSING',
                meta={
                    'progress': 0,
                    'stage': 'extracting_audio',
                    'message': '正在提取音频'
                }
            )
            
            start_time = time.time()
            
            self.update_state(
                state='PROCESSING',
                meta={
                    'progress': 50,
                    'stage': 'processing_audio',
                    'message': '正在处理音频'
                }
            )
            
            result = process_video(
                input_path,
                audio_output_path,
                output_path,
                extract_audio=True,
                do_process_audio=True,
                silence_threshold=options.get('silence_threshold', -40.0),
                min_silence_duration=options.get('min_silence_duration', 0.5),
                max_volume=options.get('max_volume', True)
            )
            
            processing_time = time.time() - start_time
            
            self.update_state(
                state='PROCESSING',
                meta={
                    'progress': 100,
                    'stage': 'completed',
                    'message': '视频处理完成',
                    'processing_time': round(processing_time, 2)
                }
            )
            
            return result
    
    except Exception as e:
        self.update_state(
            state='FAILED',
            meta={
                'error': str(e),
                'error_type': type(e).__name__,
                'stage': 'failed'
            }
        )
        
        return {
            'success': False,
            'message': f'媒体处理失败: {str(e)}',
            'error_type': type(e).__name__
        }

def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    获取任务状态
    
    Args:
        task_id: 任务ID
    
    Returns:
        任务状态字典
    """
    from celery.result import AsyncResult
    
    task = AsyncResult(task_id, app=celery_app)
    
    status_info = {
        'task_id': task_id,
        'status': task.state,
    }
    
    if task.state == 'PENDING':
        status_info['message'] = '任务等待处理'
    
    elif task.state == 'PROCESSING':
        info = task.info or {}
        status_info.update({
            'progress': info.get('progress', 0),
            'stage': info.get('stage', ''),
            'message': info.get('message', '正在处理'),
            'processing_time': info.get('processing_time', 0)
        })
    
    elif task.state == 'SUCCESS':
        status_info['result'] = task.result
        status_info['message'] = '任务完成'
    
    elif task.state == 'FAILED':
        info = task.info or {}
        status_info.update({
            'error': info.get('error', 'Unknown error'),
            'error_type': info.get('error_type', 'Unknown'),
            'message': '任务失败'
        })
    
    else:
        status_info['message'] = f'任务状态: {task.state}'
    
    return status_info

def cancel_task(task_id: str) -> bool:
    """
    取消任务
    
    Args:
        task_id: 任务ID
    
    Returns:
        是否成功取消
    """
    from celery.result import AsyncResult
    
    task = AsyncResult(task_id, app=celery_app)
    
    try:
        task.revoke(terminate=True)
        return True
    except Exception:
        return False

def get_active_tasks() -> list:
    """
    获取活跃任务列表
    
    Returns:
        活跃任务列表
    """
    inspector = celery_app.control.inspect()
    
    active_tasks = []
    
    try:
        active = inspector.active()
        if active:
            for worker, tasks in active.items():
                for task in tasks:
                    active_tasks.append({
                        'task_id': task['id'],
                        'name': task['name'],
                        'worker': worker,
                        'args': task.get('args', [])
                    })
    except Exception:
        pass
    
    return active_tasks

def get_queue_stats() -> Dict[str, int]:
    """
    获取队列统计信息
    
    Returns:
        队列统计字典
    """
    inspector = celery_app.control.inspect()
    
    stats = {
        'active_tasks': 0,
        'reserved_tasks': 0,
        'scheduled_tasks': 0
    }
    
    try:
        active = inspector.active()
        if active:
            stats['active_tasks'] = sum(len(tasks) for tasks in active.values())
        
        reserved = inspector.reserved()
        if reserved:
            stats['reserved_tasks'] = sum(len(tasks) for tasks in reserved.values())
        
        scheduled = inspector.scheduled()
        if scheduled:
            stats['scheduled_tasks'] = sum(len(tasks) for tasks in scheduled.values())
    except Exception:
        pass
    
    return stats