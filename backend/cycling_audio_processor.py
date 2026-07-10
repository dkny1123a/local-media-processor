import numpy as np
import librosa
import subprocess
import tempfile
import os
from typing import Tuple, Dict, Any


def apply_dynamic_range_compression(audio_data: np.ndarray, sample_rate: int, ratio: float = 4.0, threshold_db: float = -20.0) -> np.ndarray:
    try:
        audio_normalized = audio_data / np.max(np.abs(audio_data)) if np.max(np.abs(audio_data)) > 0 else audio_data
        
        rms = librosa.feature.rms(y=audio_normalized)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=1.0)
        
        gain_db = np.zeros_like(rms_db)
        for i, db in enumerate(rms_db):
            if db > threshold_db:
                gain_db[i] = (threshold_db - db) / ratio
        
        gain = librosa.db_to_amplitude(gain_db)
        
        frame_length = int(sample_rate * 0.05)
        hop_length = int(sample_rate * 0.025)
        
        compressed_audio = np.zeros_like(audio_normalized)
        for i in range(len(gain)):
            start = i * hop_length
            end = min(start + frame_length, len(audio_normalized))
            if start < len(audio_normalized):
                compressed_audio[start:end] = audio_normalized[start:end] * gain[i]
        
        compressed_audio = compressed_audio / np.max(np.abs(compressed_audio)) if np.max(np.abs(compressed_audio)) > 0 else compressed_audio
        
        return compressed_audio
    except Exception as e:
        print(f"动态范围压缩失败: {e}")
        return audio_data


def apply_bandpass_filter(audio_data: np.ndarray, sample_rate: int, low_cut: float = 300.0, high_cut: float = 3400.0) -> np.ndarray:
    try:
        import soundfile as sf
        
        temp_input = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_input_path = temp_input.name
        temp_input.close()
        
        sf.write(temp_input_path, audio_data, sample_rate)
        
        temp_output = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_output_path = temp_output.name
        temp_output.close()
        
        command = [
            'ffmpeg',
            '-i', temp_input_path,
            '-af', f'highpass=f={low_cut},lowpass=f={high_cut}',
            '-y',
            '-loglevel', 'quiet',
            temp_output_path
        ]
        
        subprocess.run(command, check=True, capture_output=True, timeout=120)
        
        filtered_audio, _ = librosa.load(temp_output_path, sr=sample_rate, mono=True)
        
        os.unlink(temp_input_path)
        os.unlink(temp_output_path)
        
        return filtered_audio
    except subprocess.TimeoutExpired:
        os.unlink(temp_input_path)
        os.unlink(temp_output_path)
        print(f"带通滤波超时")
    except Exception as e:
        print(f"带通滤波失败(ffmpeg): {e}")
        try:
            from scipy.signal import butter, lfilter
            
            nyquist = sample_rate / 2
            low = low_cut / nyquist
            high = high_cut / nyquist
            
            order = 4
            b, a = butter(order, [low, high], btype='band')
            
            chunk_size = int(sample_rate * 60)
            total_samples = len(audio_data)
            
            if total_samples > chunk_size:
                filtered_chunks = []
                for start in range(0, total_samples, chunk_size):
                    end = min(start + chunk_size, total_samples)
                    chunk = audio_data[start:end]
                    filtered_chunk = lfilter(b, a, chunk)
                    filtered_chunks.append(filtered_chunk)
                    del chunk
                filtered_audio = np.concatenate(filtered_chunks)
                del filtered_chunks
            else:
                filtered_audio = lfilter(b, a, audio_data)
            
            return filtered_audio
        except ImportError:
            print("警告: scipy 未安装，跳过带通滤波")
            return audio_data
        except Exception as e2:
            print(f"带通滤波失败(scipy): {e2}")
            return audio_data


def apply_voice_enhancement(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    try:
        audio_normalized = audio_data / np.max(np.abs(audio_data)) if np.max(np.abs(audio_data)) > 0 else audio_data
        
        stft = librosa.stft(audio_normalized, n_fft=512, hop_length=256)
        magnitude, phase = librosa.magphase(stft)
        
        freq_bins = librosa.fft_frequencies(sr=sample_rate, n_fft=512)
        
        voice_mask = np.zeros_like(magnitude)
        for i, freq in enumerate(freq_bins):
            if 300 <= freq <= 3400:
                voice_mask[i] = 1.0
            elif 200 <= freq < 300:
                voice_mask[i] = 0.5
            elif 3400 < freq <= 4000:
                voice_mask[i] = 0.3
        
        enhanced_magnitude = magnitude * (1 + voice_mask * 0.5)
        
        enhanced_stft = enhanced_magnitude * phase
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=256)
        
        enhanced_audio = enhanced_audio / np.max(np.abs(enhanced_audio)) if np.max(np.abs(enhanced_audio)) > 0 else enhanced_audio
        
        return enhanced_audio
    except Exception as e:
        print(f"语音增强失败: {e}")
        return audio_data


def detect_voice_activity(audio_data: np.ndarray, sample_rate: int, threshold_db: float = -45.0, min_duration: float = 0.1) -> np.ndarray:
    try:
        frame_length = int(sample_rate * 0.02)
        hop_length = int(sample_rate * 0.01)
        
        rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
        db = librosa.amplitude_to_db(rms, ref=1.0)
        
        voice_mask = db > threshold_db
        
        voice_segments = []
        in_voice = False
        start_frame = 0
        
        for j, is_voice in enumerate(voice_mask):
            if is_voice and not in_voice:
                in_voice = True
                start_frame = j
            elif not is_voice and in_voice:
                in_voice = False
                end_frame = j
                duration = (end_frame - start_frame) * hop_length / sample_rate
                if duration >= min_duration:
                    voice_segments.append((start_frame * hop_length / sample_rate, end_frame * hop_length / sample_rate))
        
        if in_voice:
            end_frame = len(voice_mask)
            duration = (end_frame - start_frame) * hop_length / sample_rate
            if duration >= min_duration:
                voice_segments.append((start_frame * hop_length / sample_rate, len(audio_data) / sample_rate))
        
        return voice_segments
    except Exception as e:
        print(f"语音活动检测失败: {e}")
        return [(0.0, len(audio_data) / sample_rate)]


def apply_intelligibility_boost(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    try:
        audio_normalized = audio_data / np.max(np.abs(audio_data)) if np.max(np.abs(audio_data)) > 0 else audio_data
        
        alpha = 1.2
        beta = 0.8
        
        frame_length = int(sample_rate * 0.02)
        hop_length = int(sample_rate * 0.01)
        
        rms = librosa.feature.rms(y=audio_normalized, frame_length=frame_length, hop_length=hop_length)[0]
        db = librosa.amplitude_to_db(rms, ref=1.0)
        
        boost_factor = np.ones_like(audio_normalized)
        
        for i in range(len(db)):
            start = i * hop_length
            end = min(start + frame_length, len(audio_normalized))
            if db[i] < -30:
                boost_factor[start:end] = alpha
            elif db[i] < -15:
                boost_factor[start:end] = beta
        
        boosted_audio = audio_normalized * boost_factor
        boosted_audio = boosted_audio / np.max(np.abs(boosted_audio)) if np.max(np.abs(boosted_audio)) > 0 else boosted_audio
        
        return boosted_audio
    except Exception as e:
        print(f"清晰度增强失败: {e}")
        return audio_data


def apply_bluetooth_optimization(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    try:
        audio_normalized = audio_data / np.max(np.abs(audio_data)) if np.max(np.abs(audio_data)) > 0 else audio_data
        
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            audio_normalized = librosa.resample(audio_normalized, orig_sr=sample_rate, target_sr=target_sample_rate)
            sample_rate = target_sample_rate
        
        return audio_normalized, sample_rate
    except Exception as e:
        print(f"蓝牙优化失败: {e}")
        return audio_data, sample_rate


def process_single_cycling_chunk(audio_chunk, sample_rate, noise_reduction, 
                                 silence_threshold, min_silence_duration, 
                                 highpass_cutoff):
    silence_count = 0
    try:
        audio_chunk = apply_bandpass_filter(audio_chunk, sample_rate)
        audio_chunk = apply_voice_enhancement(audio_chunk, sample_rate)
        audio_chunk = apply_dynamic_range_compression(audio_chunk, sample_rate)
        audio_chunk = apply_intelligibility_boost(audio_chunk, sample_rate)
        
        if noise_reduction > 0:
            try:
                import noisereduce as nr
                reduced_chunk = nr.reduce_noise(
                    y=audio_chunk,
                    sr=sample_rate,
                    prop_decrease=noise_reduction
                )
                audio_chunk = audio_chunk * (1 - noise_reduction) + reduced_chunk * noise_reduction
            except Exception:
                pass
        
        voice_segments = detect_voice_activity(audio_chunk, sample_rate, silence_threshold, min_silence_duration)
        
        if len(voice_segments) > 0:
            result_segments = []
            soft_boundary_samples = int(sample_rate * 50 / 1000)
            
            for start, end in voice_segments:
                start_sample = int(start * sample_rate)
                end_sample = int(end * sample_rate)
                
                segment = audio_chunk[start_sample:end_sample]
                
                if len(segment) > soft_boundary_samples * 2:
                    fade_in = np.linspace(0, 1, soft_boundary_samples)
                    fade_out = np.linspace(1, 0, soft_boundary_samples)
                    
                    segment[:soft_boundary_samples] = segment[:soft_boundary_samples] * fade_in
                    segment[-soft_boundary_samples:] = segment[-soft_boundary_samples:] * fade_out
                
                result_segments.append(segment)
            
            audio_chunk = np.concatenate(result_segments)
            
            total_voice_duration = sum(end - start for start, end in voice_segments)
            total_duration = len(audio_chunk) / sample_rate
            silence_count = len(voice_segments) - 1 if len(voice_segments) > 1 else 0
        
        return audio_chunk, silence_count
    except Exception as e:
        print(f"分块处理失败: {e}")
        return audio_chunk, 0


def process_cycling_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    noise_reduction: float = 0.7,
    silence_threshold: float = -45.0,
    min_silence_duration: float = 0.3,
    max_volume: bool = True,
    target_db: float = -3.0,
    highpass_cutoff: float = 100.0,
    progress_callback=None
) -> Tuple[np.ndarray, Dict[str, Any], int]:
    def update_progress(pct, msg, status=None):
        if progress_callback:
            progress_callback(pct, msg, status)
    
    stats = {
        'original_duration': round(len(audio_data) / sample_rate, 2),
        'processed_duration': 0,
        'silence_segments_removed': 0
    }
    
    total_samples = len(audio_data)
    chunk_duration = 60
    chunk_size = int(sample_rate * chunk_duration)
    num_chunks = int(np.ceil(total_samples / chunk_size))
    
    if num_chunks < 1:
        num_chunks = 1
    
    if num_chunks >= 1:
        update_progress(5, f'分块处理({num_chunks}块)...', 'processing')
        
        processed_chunks = []
        total_silence_removed = 0
        last_update_pct = 0
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, total_samples)
            chunk = audio_data[start:end]
            
            progress_pct = 5 + int((i / num_chunks) * 50)
            
            if progress_pct > last_update_pct or i % 5 == 0 or i == num_chunks - 1:
                update_progress(progress_pct, f'处理块 {i+1}/{num_chunks}...', 'processing')
                last_update_pct = progress_pct
            
            processed_chunk, silence_count = process_single_cycling_chunk(chunk, sample_rate, noise_reduction,
                                                                          silence_threshold, min_silence_duration,
                                                                          highpass_cutoff)
            processed_chunks.append(processed_chunk)
            total_silence_removed += silence_count
            
            del chunk
            import gc
            gc.collect()
        
        audio_data = np.concatenate(processed_chunks)
        del processed_chunks
        gc.collect()
        
        stats['silence_segments_removed'] = total_silence_removed
    
    update_progress(80, '蓝牙优化（降采样至16kHz）...', 'processing')
    audio_data, sample_rate = apply_bluetooth_optimization(audio_data, sample_rate)
    
    update_progress(90, '音量归一化...', 'processing')
    if max_volume:
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_path = temp_wav.name
        temp_wav.close()
        
        import soundfile as sf
        sf.write(temp_path, audio_data, sample_rate)
        
        normalized_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        normalized_wav.close()
        normalized_path = normalized_wav.name
        
        command = [
            'ffmpeg',
            '-i', temp_path,
            '-af', f'loudnorm=I={target_db}:LRA=11:TP=-1.5',
            '-y',
            '-loglevel', 'quiet',
            normalized_path
        ]
        
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=300)
            audio_data, _ = librosa.load(normalized_path, sr=sample_rate, mono=True)
            os.unlink(temp_path)
            os.unlink(normalized_path)
        except subprocess.TimeoutExpired:
            print(f"音量归一化超时")
            os.unlink(temp_path)
            os.unlink(normalized_path)
        except:
            os.unlink(temp_path)
            os.unlink(normalized_path)
    
    stats['processed_duration'] = round(len(audio_data) / sample_rate, 2)
    
    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_path = temp_wav.name
    temp_wav.close()
    
    import soundfile as sf
    sf.write(temp_path, audio_data, sample_rate)
    
    return audio_data, stats, temp_path