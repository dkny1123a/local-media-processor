import librosa
import soundfile as sf
import numpy as np
import subprocess
import os
import tempfile
import uuid

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

def detect_silence(audio_data, sample_rate, threshold=-40.0, min_silence_duration=0.5):
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    frame_length = int(sample_rate * 0.02)
    hop_length = int(sample_rate * 0.01)
    
    rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)
    db = librosa.amplitude_to_db(rms, ref=np.max)
    
    silence_mask = db < threshold
    
    silence_segments = []
    in_silence = False
    start_frame = 0
    
    for i, is_silent in enumerate(silence_mask[0]):
        if is_silent and not in_silence:
            in_silence = True
            start_frame = i
        elif not is_silent and in_silence:
            in_silence = False
            end_frame = i
            duration = (end_frame - start_frame) * hop_length / sample_rate
            if duration >= min_silence_duration:
                start_time = start_frame * hop_length / sample_rate
                end_time = end_frame * hop_length / sample_rate
                silence_segments.append((start_time, end_time))
    
    if in_silence:
        end_frame = len(silence_mask[0])
        duration = (end_frame - start_frame) * hop_length / sample_rate
        if duration >= min_silence_duration:
            start_time = start_frame * hop_length / sample_rate
            end_time = len(audio_data) / sample_rate
            silence_segments.append((start_time, end_time))
    
    return silence_segments

def remove_silence(audio_data, sample_rate, silence_segments):
    if not silence_segments:
        return audio_data
    
    segments_to_keep = []
    last_end = 0
    
    for start, end in sorted(silence_segments):
        if start > last_end:
            segments_to_keep.append((last_end, start))
        last_end = end
    
    if last_end < len(audio_data) / sample_rate:
        segments_to_keep.append((last_end, len(audio_data) / sample_rate))
    
    if not segments_to_keep:
        return audio_data
    
    result = []
    for start, end in segments_to_keep:
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        result.append(audio_data[start_sample:end_sample])
    
    return np.concatenate(result)

def normalize_volume(audio_data, target_db=-1.0):
    rms = np.sqrt(np.mean(audio_data**2))
    if rms == 0:
        return audio_data
    
    current_db = 20 * np.log10(rms)
    gain = 10 ** ((target_db - current_db) / 20)
    result = audio_data * gain
    
    max_val = np.max(np.abs(result))
    if max_val > 1.0:
        result = result / max_val
    
    return result

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

def apply_denoise(audio_data, sample_rate, noise_reduction=0.0, stationary_noise=False):
    if noise_reduction <= 0:
        return audio_data
    
    try:
        import noisereduce as nr
        
        if stationary_noise:
            noise_sample = audio_data[:int(sample_rate * 0.5)]
            reduced_noise = nr.reduce_noise(
                y=audio_data,
                sr=sample_rate,
                y_noise=noise_sample,
                verbose=False
            )
        else:
            reduced_noise = nr.reduce_noise(
                y=audio_data,
                sr=sample_rate,
                verbose=False
            )
        
        alpha = noise_reduction
        result = audio_data * (1 - alpha) + reduced_noise * alpha
        
        return result
    except ImportError:
        print("noisereduce 模块未安装，跳过降噪")
        return audio_data
    except Exception as e:
        print(f"降噪处理失败: {str(e)}")
        return audio_data

def process_audio(input_path, output_path, silence_threshold=-40.0, min_silence_duration=0.5, max_volume=True, noise_reduction=0.0, stationary_noise=False):
    try:
        output_path = os.path.splitext(output_path)[0] + '.mp3'
        
        converted_path, was_converted = convert_to_wav(input_path)
        
        audio_data, sample_rate = librosa.load(converted_path, sr=None, mono=True)
        
        if was_converted:
            os.unlink(converted_path)
        
        original_duration = len(audio_data) / sample_rate
        
        audio_data = apply_denoise(audio_data, sample_rate, noise_reduction, stationary_noise)
        
        silence_segments = detect_silence(audio_data, sample_rate, silence_threshold, min_silence_duration)
        
        processed_audio = remove_silence(audio_data, sample_rate, silence_segments)
        
        if max_volume:
            processed_audio = normalize_volume(processed_audio)
        
        success = save_as_mp3(processed_audio, sample_rate, output_path)
        
        if not success:
            return {
                "success": False,
                "message": "音频处理失败: 无法保存为MP3格式"
            }
        
        processed_duration = len(processed_audio) / sample_rate
        silence_removed = len(silence_segments)
        duration_reduction = ((original_duration - processed_duration) / original_duration * 100) if original_duration > 0 else 0
        
        return {
            "success": True,
            "message": "音频处理完成",
            "stats": {
                "original_duration": round(original_duration, 2),
                "processed_duration": round(processed_duration, 2),
                "silence_segments_removed": silence_removed,
                "duration_reduction_percent": round(duration_reduction, 2),
                "noise_reduction": noise_reduction,
                "sample_rate": sample_rate
            }
        }
    
    except Exception as e:
        return {
            "success": False,
            "message": f"音频处理失败: {str(e)}"
        }