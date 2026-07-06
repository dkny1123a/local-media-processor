"""
优化后的 Demucs 处理器 - 使用模型管理器
支持音源分离、人声提取等功能
"""
import torch
import os
import time
import logging
from typing import Dict, Any, Optional
from demucs import pretrained, separate
from demucs.audio import AudioFile

logger = logging.getLogger(__name__)

# 导入模型管理器
from .model_manager import model_manager, ModelPriority

# ==================== 模型加载函数 ====================

def get_demucs_model(model_name: str = "htdemucs") -> Any:
    """
    通过模型管理器加载 Demucs 模型
    
    Args:
        model_name: 模型名称 (htdemucs, htdemucs_ft, mdx, mdx_extra, mdx_q)
    
    Returns:
        Demucs 模型实例
    """
    return model_manager.load_model(
        'demucs',
        model_name,
        lambda name: pretrained.get_model(name),
        priority=ModelPriority.LOW  # Demucs 模型低优先级，容易释放
    )

def get_available_models() -> list:
    """
    获取可用的 Demucs 模型列表
    
    Returns:
        模型名称列表
    """
    return ["htdemucs", "htdemucs_ft", "mdx", "mdx_extra", "mdx_q"]

# ==================== 音源分离函数 ====================

def separate_audio(
    audio_path: str,
    output_dir: str,
    model_name: str = "htdemucs",
    two_stems: bool = False
) -> Dict[str, Any]:
    """
    使用 Demucs 模型分离音源
    
    Args:
        audio_path: 音频文件路径
        output_dir: 输出目录
        model_name: 模型名称
        two_stems: 是否只分离人声和伴奏
    
    Returns:
        分离结果字典
    """
    start_time = time.time()
    
    try:
        logger.info(f"开始音源分离: {audio_path}, model={model_name}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 通过模型管理器加载模型
        model = get_demucs_model(model_name)
        
        # 加载音频
        audio_file = AudioFile(audio_path)
        audio = audio_file.read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        audio = audio.to(device)
        
        # 执行分离
        sources = separate.separate(model, audio, shifts=1, split=True, overlap=0.25)
        
        # 保存分离结果
        source_names = model.sources
        separated_files = []
        
        for i, source_name in enumerate(source_names):
            source_audio = sources[i]
            
            # 保存音频文件
            output_path = os.path.join(output_dir, f"{source_name}.wav")
            
            # 转换为 numpy 并保存
            import soundfile as sf
            sf.write(output_path, source_audio.cpu().numpy().T, model.samplerate)
            
            separated_files.append({
                "name": source_name,
                "path": output_path,
                "size_mb": os.path.getsize(output_path) / 1024 / 1024
            })
        
        processing_time = time.time() - start_time
        
        logger.info(f"音源分离完成: {audio_path}, time={processing_time:.2f}s")
        
        # 返回结果
        return {
            "success": True,
            "model": model_name,
            "sources": separated_files,
            "output_dir": output_dir,
            "stats": {
                "processing_time": round(processing_time, 2),
                "sources_count": len(separated_files),
                "total_size_mb": sum(f["size_mb"] for f in separated_files)
            }
        }
    
    except Exception as e:
        logger.error(f"音源分离失败: {audio_path}, error={str(e)}")
        
        return {
            "success": False,
            "message": f"音源分离失败: {str(e)}",
            "error_type": type(e).__name__
        }

def separate_vocals_only(
    audio_path: str,
    output_dir: str,
    model_name: str = "htdemucs"
) -> Dict[str, Any]:
    """
    只分离人声和伴奏
    
    Args:
        audio_path: 音频文件路径
        output_dir: 输出目录
        model_name: 模型名称
    
    Returns:
        分离结果字典
    """
    start_time = time.time()
    
    try:
        logger.info(f"开始人声分离: {audio_path}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 通过模型管理器加载模型
        model = get_demucs_model(model_name)
        
        # 加载音频
        audio_file = AudioFile(audio_path)
        audio = audio_file.read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        audio = audio.to(device)
        
        # 执行分离
        sources = separate.separate(model, audio, shifts=1, split=True, overlap=0.25)
        
        # 获取人声和伴奏
        source_names = model.sources
        
        # 保存人声和伴奏
        separated_files = []
        
        for i, source_name in enumerate(source_names):
            if source_name in ['vocals', 'drums', 'bass', 'other']:
                source_audio = sources[i]
                
                output_path = os.path.join(output_dir, f"{source_name}.wav")
                
                import soundfile as sf
                sf.write(output_path, source_audio.cpu().numpy().T, model.samplerate)
                
                separated_files.append({
                    "name": source_name,
                    "path": output_path,
                    "size_mb": os.path.getsize(output_path) / 1024 / 1024
                })
        
        processing_time = time.time() - start_time
        
        logger.info(f"人声分离完成: {audio_path}, time={processing_time:.2f}s")
        
        return {
            "success": True,
            "model": model_name,
            "sources": separated_files,
            "output_dir": output_dir,
            "stats": {
                "processing_time": round(processing_time, 2),
                "sources_count": len(separated_files)
            }
        }
    
    except Exception as e:
        logger.error(f"人声分离失败: {audio_path}, error={str(e)}")
        
        return {
            "success": False,
            "message": f"人声分离失败: {str(e)}",
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
    model_info = {
        "htdemucs": {"size_mb": 1000, "sources": ["drums", "bass", "other", "vocals"]},
        "htdemucs_ft": {"size_mb": 1000, "sources": ["drums", "bass", "other", "vocals"]},
        "mdx": {"size_mb": 500, "sources": ["drums", "bass", "other", "vocals"]},
        "mdx_extra": {"size_mb": 800, "sources": ["drums", "bass", "other", "vocals"]}
    }
    
    info = model_info.get(model_name, {})
    
    # 添加模型管理器信息
    if model_manager.has_model('demucs', model_name):
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
    return model_manager.release_model('demucs', model_name)