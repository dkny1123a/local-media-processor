import librosa
import soundfile as sf
import numpy as np
import tempfile
import os
import subprocess
from typing import Dict, Any, Optional

def convert_to_wav(input_path):
    ext = os.path.splitext(input_path)[1].lower()
    
    amr_formats = ['.amr', '.3gp']
    m4a_formats = ['.m4a']
    
    if ext in amr_formats or ext in m4a_formats:
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_path = temp_wav.name
        temp_wav.close()
        
        command = [
            'ffmpeg',
            '-i', input_path,
            '-ac', '1',
            '-ar', '44100',
            '-y',
            temp_path
        ]
        
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=120)
            return temp_path, True
        except subprocess.CalledProcessError as e:
            print(f"格式转换失败: {e.stderr.decode()}")
            return input_path, False
        except subprocess.TimeoutExpired:
            print(f"格式转换超时")
            os.unlink(temp_path)
            return input_path, False
    
    return input_path, False

def denoise_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    reduction_amount: float = 0.5,
    stationary_noise: bool = False,
    n_fft: int = 512,
    win_length: int = 512,
    hop_length: int = 128
) -> np.ndarray:
    try:
        import noisereduce as nr
        
        if stationary_noise:
            noise_sample = audio_data[:int(sample_rate * 0.5)]
            reduced_noise = nr.reduce_noise(
                y=audio_data,
                sr=sample_rate,
                y_noise=noise_sample,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                verbose=False
            )
        else:
            reduced_noise = nr.reduce_noise(
                y=audio_data,
                sr=sample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                verbose=False
            )
        
        alpha = reduction_amount
        result = audio_data * (1 - alpha) + reduced_noise * alpha
        
        return result
    
    except ImportError:
        print("noisereduce 模块未安装，跳过降噪")
        return audio_data
    except Exception as e:
        print(f"降噪处理失败: {str(e)}")
        return audio_data

def enhance_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    noise_reduction: float = 0.5,
    stationary_noise: bool = False,
    normalize_volume: bool = True,
    target_db: float = -1.0,
    n_fft: int = 512
) -> np.ndarray:
    if noise_reduction > 0:
        audio_data = denoise_audio(
            audio_data,
            sample_rate,
            reduction_amount=noise_reduction,
            stationary_noise=stationary_noise,
            n_fft=n_fft
        )
    
    if normalize_volume:
        rms = np.sqrt(np.mean(audio_data**2))
        if rms > 0:
            current_db = 20 * np.log10(rms)
            gain = 10 ** ((target_db - current_db) / 20)
            audio_data = audio_data * gain
            
            max_val = np.max(np.abs(audio_data))
            if max_val > 1.0:
                audio_data = audio_data / max_val
    
    return audio_data

def save_as_mp3(audio_data, sample_rate, output_path):
    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_path = temp_wav.name
    temp_wav.close()
    
    try:
        sf.write(temp_path, audio_data, sample_rate)
        
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        command = [
            'ffmpeg',
            '-i', temp_path,
            '-ac', '1',
            '-ar', '44100',
            '-c:a', 'libmp3lame',
            '-q:a', '2',
            '-y',
            output_path
        ]
        
        subprocess.run(command, check=True, capture_output=True, timeout=600)
        return True
    except subprocess.TimeoutExpired:
        print(f"保存MP3超时")
        return False
    except Exception as e:
        print(f"保存MP3失败: {str(e)}")
        return False
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def process_denoise(
    input_path: str,
    output_path: str,
    noise_reduction: float = 0.5,
    stationary_noise: bool = False,
    normalize_volume: bool = True,
    target_db: float = -1.0
) -> Dict[str, Any]:
    try:
        output_path = os.path.splitext(output_path)[0] + '.mp3'
        
        converted_path, was_converted = convert_to_wav(input_path)
        
        audio_data, sample_rate = librosa.load(converted_path, sr=None, mono=True)
        
        if was_converted:
            os.unlink(converted_path)
        
        original_duration = len(audio_data) / sample_rate
        original_rms = np.sqrt(np.mean(audio_data**2))
        
        processed_audio = enhance_audio(
            audio_data,
            sample_rate,
            noise_reduction=noise_reduction,
            stationary_noise=stationary_noise,
            normalize_volume=normalize_volume,
            target_db=target_db
        )
        
        processed_rms = np.sqrt(np.mean(processed_audio**2))
        
        success = save_as_mp3(processed_audio, sample_rate, output_path)
        
        if not success:
            return {
                "success": False,
                "message": "降噪处理失败: 无法保存为MP3格式"
            }
        
        processed_duration = len(processed_audio) / sample_rate
        
        return {
            "success": True,
            "message": "降噪增强处理完成",
            "stats": {
                "original_duration": round(original_duration, 2),
                "processed_duration": round(processed_duration, 2),
                "sample_rate": sample_rate,
                "noise_reduction": noise_reduction,
                "normalized": normalize_volume,
                "original_rms": round(original_rms, 6),
                "processed_rms": round(processed_rms, 6)
            }
        }
    
    except Exception as e:
        return {
            "success": False,
            "message": f"降噪处理失败: {str(e)}"
        }
