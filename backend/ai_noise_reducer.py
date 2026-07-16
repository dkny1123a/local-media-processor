import os
import numpy as np
import tempfile
import soundfile as sf
import librosa

_frcrn_pipeline = None
_frcrn_initialized = False


def init_frcrn(model_dir=None):
    global _frcrn_pipeline, _frcrn_initialized
    
    if _frcrn_initialized:
        return _frcrn_pipeline is not None
    
    _frcrn_initialized = True
    
    try:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        
        if model_dir:
            os.environ['MODELSCOPE_CACHE'] = model_dir
        
        try:
            import onnxruntime
            _frcrn_pipeline = pipeline(
                Tasks.acoustic_noise_suppression,
                model='damo/speech_frcrn_ans_cirm_16k',
                device='cpu',
            )
            print("[FRCRN] 模型加载成功 (ONNX)")
        except ImportError:
            _frcrn_pipeline = pipeline(
                Tasks.acoustic_noise_suppression,
                model='damo/speech_frcrn_ans_cirm_16k',
            )
            print("[FRCRN] 模型加载成功 (PyTorch)")
        
        return True
    except Exception as e:
        print(f"[FRCRN] 模型加载失败: {e}")
        _frcrn_pipeline = None
        return False


def get_frcrn_pipeline():
    global _frcrn_pipeline, _frcrn_initialized
    if not _frcrn_initialized:
        init_frcrn()
    return _frcrn_pipeline


def analyze_noise_profile(audio_data: np.ndarray, sample_rate: int) -> dict:
    from .core.audio_analyzer import analyze_audio_spectrum
    return analyze_audio_spectrum(audio_data, sample_rate)


def frcrn_denoise_chunk(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    pipeline = get_frcrn_pipeline()
    
    if pipeline is None:
        return audio_data
    
    try:
        if sample_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        temp_input = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_input_path = temp_input.name
        temp_input.close()
        
        sf.write(temp_input_path, audio_data, sample_rate, subtype='PCM_16')
        
        temp_output = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_output_path = temp_output.name
        temp_output.close()
        
        pipeline(temp_input_path, output_path=temp_output_path)
        
        denoised_audio, _ = librosa.load(temp_output_path, sr=sample_rate, mono=True)
        
        os.unlink(temp_input_path)
        os.unlink(temp_output_path)
        
        return denoised_audio
    
    except Exception as e:
        print(f"[FRCRN] 降噪失败: {e}")
        return audio_data


def adaptive_denoise(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    profile = analyze_noise_profile(audio_data, sample_rate)
    
    if profile['noise_level'] == 'high':
        print(f"[FRCRN] 高噪声环境(分数={profile['noise_score']:.2f})，使用FRCRN降噪")
        return frcrn_denoise_chunk(audio_data, sample_rate)
    else:
        print(f"[FRCRN] 低/中等噪声环境(分数={profile['noise_score']:.2f})，使用传统降噪")
        return audio_data