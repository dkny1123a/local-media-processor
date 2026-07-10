import os
import numpy as np

_vad_model = None
_vad_initialized = False


def init_vad(model_dir=None):
    global _vad_model, _vad_initialized
    
    if _vad_initialized:
        return _vad_model is not None
    
    _vad_initialized = True
    
    try:
        from silero_vad import load_silero_vad
        try:
            import onnxruntime as ort
            _vad_model = load_silero_vad(onnx=True)
            print("[VAD] Silero VAD模型加载成功 (ONNX)")
            return True
        except ImportError:
            _vad_model = load_silero_vad()
            print("[VAD] Silero VAD模型加载成功 (JIT)")
            return True
        except Exception as e:
            _vad_model = load_silero_vad()
            print(f"[VAD] Silero VAD模型加载成功 (JIT, ONNX失败: {e})")
            return True
    except Exception as e:
        print(f"[VAD] 模型加载失败: {e}")
        _vad_model = None
        return False


def get_vad_model():
    global _vad_model, _vad_initialized
    if not _vad_initialized:
        init_vad()
    return _vad_model


def detect_voice_segments(audio_data, sample_rate, min_speech_duration=0.3, min_silence_duration=0.5, 
                          threshold=0.5):
    vad_model = get_vad_model()
    
    if vad_model is None:
        return [(0.0, len(audio_data) / sample_rate)]
    
    try:
        from silero_vad import get_speech_timestamps
        import torch
        
        if sample_rate not in [8000, 16000]:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0 and max_val > 0:
            audio_data = audio_data / max_val
        
        torch.set_num_threads(1)
        audio_tensor = torch.from_numpy(audio_data)
        
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            vad_model,
            threshold=threshold,
            min_silence_duration_ms=int(min_silence_duration * 1000),
            min_speech_duration_ms=int(min_speech_duration * 1000),
            return_seconds=True
        )
        
        segments = [(s['start'], s['end']) for s in speech_timestamps]
        
        return segments
    
    except Exception as e:
        print(f"[VAD] 检测失败: {e}")
        return [(0.0, len(audio_data) / sample_rate)]


def detect_non_voice_segments_vad(audio_data, sample_rate, min_duration=0.5):
    voice_segments = detect_voice_segments(audio_data, sample_rate, 
                                           min_speech_duration=0.3,
                                           min_silence_duration=min_duration)
    
    total_duration = len(audio_data) / sample_rate
    non_voice_segments = []
    last_end = 0.0
    
    for start, end in sorted(voice_segments):
        if start > last_end + 0.01:
            non_voice_duration = start - last_end
            if non_voice_duration >= min_duration:
                non_voice_segments.append((last_end, start))
        last_end = end
    
    if last_end < total_duration - 0.01:
        non_voice_duration = total_duration - last_end
        if non_voice_duration >= min_duration:
            non_voice_segments.append((last_end, total_duration))
    
    return non_voice_segments
