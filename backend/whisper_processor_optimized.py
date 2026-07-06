"""
优化后的 Whisper 处理器 - 使用模型管理器
支持语音识别、语言检测、转录等功能
"""
import whisper
import os
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# 导入模型管理器
from .model_manager import model_manager, ModelPriority

# ==================== 模型加载函数 ====================

def get_model(model_name: str = "large-v3") -> Any:
    """
    通过模型管理器加载 Whisper 模型
    
    Args:
        model_name: 模型名称 (tiny, base, small, medium, large, large-v3)
    
    Returns:
        Whisper 模型实例
    """
    return model_manager.load_model(
        'whisper',
        model_name,
        lambda name: whisper.load_model(name),
        priority=ModelPriority.MEDIUM  # Whisper 模型中等优先级
    )

def get_available_models() -> list:
    """
    获取可用的 Whisper 模型列表
    
    Returns:
        模型名称列表
    """
    return ["tiny", "base", "small", "medium", "large", "large-v3"]

# ==================== 语音识别函数 ====================

def transcribe_audio(
    audio_path: str,
    model_name: str = "large-v3",
    language: Optional[str] = None,
    task: str = "transcribe"
) -> Dict[str, Any]:
    """
    使用 Whisper 模型转录音频
    
    Args:
        audio_path: 音频文件路径
        model_name: 模型名称
        language: 语言代码（可选，自动检测）
        task: 任务类型 (transcribe/translate)
    
    Returns:
        转录结果字典
    """
    start_time = time.time()
    
    try:
        logger.info(f"开始语音识别: {audio_path}, model={model_name}")
        
        # 通过模型管理器加载模型
        model = get_model(model_name)
        
        # 执行转录
        result = model.transcribe(
            audio_path,
            language=language,
            task=task,
            verbose=False
        )
        
        processing_time = time.time() - start_time
        
        logger.info(f"语音识别完成: {audio_path}, time={processing_time:.2f}s")
        
        # 返回结果
        return {
            "success": True,
            "text": result["text"],
            "segments": result.get("segments", []),
            "language": result.get("language", "unknown"),
            "model": model_name,
            "stats": {
                "processing_time": round(processing_time, 2),
                "audio_duration": result.get("segments", [{}])[-1].get("end", 0) if result.get("segments") else 0,
                "segments_count": len(result.get("segments", []))
            }
        }
    
    except Exception as e:
        logger.error(f"语音识别失败: {audio_path}, error={str(e)}")
        
        return {
            "success": False,
            "message": f"语音识别失败: {str(e)}",
            "error_type": type(e).__name__
        }

def detect_language(audio_path: str, model_name: str = "base") -> Dict[str, Any]:
    """
    检测音频语言
    
    Args:
        audio_path: 音频文件路径
        model_name: 模型名称（使用较小的模型）
    
    Returns:
        语言检测结果
    """
    try:
        logger.info(f"检测语言: {audio_path}")
        
        # 使用较小的模型检测语言
        model = get_model(model_name)
        
        # 加载音频
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        
        # 创建 log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        
        # 检测语言
        _, probs = model.detect_language(mel)
        detected_language = max(probs, key=probs.get)
        
        logger.info(f"检测到语言: {detected_language}")
        
        return {
            "success": True,
            "language": detected_language,
            "language_probs": probs,
            "model": model_name
        }
    
    except Exception as e:
        logger.error(f"语言检测失败: {audio_path}, error={str(e)}")
        
        return {
            "success": False,
            "message": f"语言检测失败: {str(e)}",
            "error_type": type(e).__name__
        }

# ==================== 辅助函数 ====================

def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    获取模型信息
    
    Args:
        model_name: 模型名称
    
    Returns:
        模型信息字典
    """
    model_sizes = {
        "tiny": {"size_mb": 39, "params": "39M", "speed": "~32x"},
        "base": {"size_mb": 74, "params": "74M", "speed": "~16x"},
        "small": {"size_mb": 244, "params": "244M", "speed": "~6x"},
        "medium": {"size_mb": 769, "params": "769M", "speed": "~2x"},
        "large": {"size_mb": 1550, "params": "1550M", "speed": "~1x"},
        "large-v3": {"size_mb": 2900, "params": "2900M", "speed": "~1x"}
    }
    
    info = model_sizes.get(model_name, {})
    
    # 添加模型管理器信息
    if model_manager.has_model('whisper', model_name):
        info["loaded"] = True
        info["memory_mb"] = model_manager.get_memory_stats().get('current_memory_mb', 0)
    else:
        info["loaded"] = False
    
    return info

def release_model(model_name: str) -> bool:
    """
    释放模型
    
    Args:
        model_name: 模型名称
    
    Returns:
        是否成功释放
    """
    return model_manager.release_model('whisper', model_name)

# ==================== 批量处理函数 ====================

def batch_transcribe(
    audio_paths: list,
    model_name: str = "large-v3",
    language: Optional[str] = None
) -> list:
    """
    批量转录音频
    
    Args:
        audio_paths: 音频文件路径列表
        model_name: 模型名称
        language: 语言代码
    
    Returns:
        转录结果列表
    """
    results = []
    
    # 加载模型一次（模型管理器会缓存）
    model = get_model(model_name)
    
    for audio_path in audio_paths:
        result = transcribe_audio(audio_path, model_name, language)
        results.append(result)
    
    return results