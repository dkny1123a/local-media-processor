import numpy as np
import librosa
from typing import Dict, Any, Tuple


def apply_highpass_filter(
    audio_data: np.ndarray,
    sample_rate: int,
    cutoff_freq: float = 100.0
) -> np.ndarray:
    nyquist = sample_rate / 2
    if cutoff_freq >= nyquist:
        return audio_data
    
    try:
        from scipy.signal import butter, lfilter
        
        order = 4
        b, a = butter(order, cutoff_freq / nyquist, btype='high')
        
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
        print("警告: scipy 未安装，跳过高通滤波")
        return audio_data
    except Exception as e:
        print(f"高通滤波失败: {e}，跳过高通滤波")
        return audio_data


def analyze_audio_characteristics(
    audio_data: np.ndarray,
    sample_rate: int
) -> Dict[str, Any]:
    frame_length = int(sample_rate * 0.02)
    hop_length = int(sample_rate * 0.01)
    
    total_samples = len(audio_data)
    analysis_duration = 60
    analysis_samples = int(sample_rate * analysis_duration)
    
    all_rms = []
    all_db = []
    num_segments = min(10, max(1, total_samples // analysis_samples))
    
    for i in range(num_segments):
        offset = int(i * (total_samples - analysis_samples) / max(num_segments - 1, 1)) if num_segments > 1 else 0
        segment = audio_data[offset:offset + analysis_samples]
        
        if len(segment) == 0:
            continue
        
        rms = librosa.feature.rms(y=segment, frame_length=frame_length, hop_length=hop_length)[0]
        all_rms.extend(rms)
    
    if len(all_rms) == 0:
        return {
            'rms_mean': 0.0,
            'rms_median': 0.0,
            'rms_std': 0.0,
            'rms_min': 0.0,
            'rms_max': 0.0,
            'noise_floor_db': -80.0,
            'signal_peak_db': -80.0,
            'signal_to_noise_ratio': 0.0,
            'dynamic_range': 0.0,
            'rms_coefficient_of_variation': 0.0,
            'silence_ratio_50db': 0.0,
            'silence_ratio_45db': 0.0,
            'silence_ratio_40db': 0.0,
            'avg_quiet_frame_db': 0.0,
            'avg_loud_frame_db': 0.0
        }
    
    rms_array = np.array(all_rms)
    
    rms_mean = np.mean(rms_array)
    rms_median = np.median(rms_array)
    rms_std = np.std(rms_array)
    rms_min = np.min(rms_array)
    rms_max = np.max(rms_array)
    
    rms_db = librosa.amplitude_to_db(rms_array, ref=1.0)
    
    noise_floor_db = np.percentile(rms_db, 3)
    signal_peak_db = np.percentile(rms_db, 97)
    
    dynamic_range = signal_peak_db - noise_floor_db
    
    rms_coefficient_of_variation = rms_std / rms_mean if rms_mean > 0 else 0.0
    
    snr_raw = signal_peak_db - noise_floor_db
    
    quiet_frames = rms_db[rms_db < np.percentile(rms_db, 20)]
    loud_frames = rms_db[rms_db > np.percentile(rms_db, 80)]
    avg_quiet_frame_db = np.mean(quiet_frames) if len(quiet_frames) > 0 else noise_floor_db
    avg_loud_frame_db = np.mean(loud_frames) if len(loud_frames) > 0 else signal_peak_db
    
    silence_ratio_50db = float(np.mean(rms_db < -50))
    silence_ratio_45db = float(np.mean(rms_db < -45))
    silence_ratio_40db = float(np.mean(rms_db < -40))
    
    return {
        'rms_mean': float(rms_mean),
        'rms_median': float(rms_median),
        'rms_std': float(rms_std),
        'rms_min': float(rms_min),
        'rms_max': float(rms_max),
        'noise_floor_db': float(noise_floor_db),
        'signal_peak_db': float(signal_peak_db),
        'signal_to_noise_ratio': float(snr_raw),
        'dynamic_range': float(dynamic_range),
        'rms_coefficient_of_variation': float(rms_coefficient_of_variation),
        'silence_ratio_50db': silence_ratio_50db,
        'silence_ratio_45db': silence_ratio_45db,
        'silence_ratio_40db': silence_ratio_40db,
        'avg_quiet_frame_db': float(avg_quiet_frame_db),
        'avg_loud_frame_db': float(avg_loud_frame_db)
    }


def calculate_adaptive_parameters(
    analysis: Dict[str, Any],
    scene: str = None
) -> Dict[str, float]:
    noise_floor_db = analysis['noise_floor_db']
    signal_peak_db = analysis['signal_peak_db']
    signal_to_noise_ratio = analysis['signal_to_noise_ratio']
    dynamic_range = analysis['dynamic_range']
    rms_coefficient_of_variation = analysis['rms_coefficient_of_variation']
    silence_ratio_50db = analysis['silence_ratio_50db']
    silence_ratio_45db = analysis['silence_ratio_45db']
    silence_ratio_40db = analysis['silence_ratio_40db']
    avg_quiet_frame_db = analysis['avg_quiet_frame_db']
    avg_loud_frame_db = analysis['avg_loud_frame_db']
    
    if dynamic_range < 5:
        silence_threshold_db = noise_floor_db + 3
    elif dynamic_range < 15:
        silence_threshold_db = noise_floor_db + (signal_peak_db - noise_floor_db) * 0.25
    elif dynamic_range < 25:
        silence_threshold_db = noise_floor_db + (signal_peak_db - noise_floor_db) * 0.30
    else:
        silence_threshold_db = noise_floor_db + (signal_peak_db - noise_floor_db) * 0.35
    
    silence_threshold_db = max(silence_threshold_db, noise_floor_db + 2)
    silence_threshold_db = min(silence_threshold_db, signal_peak_db - 5)
    
    if signal_to_noise_ratio < 5:
        noise_reduction = 0.95
    elif signal_to_noise_ratio < 10:
        noise_reduction = 0.85 + (signal_to_noise_ratio - 5) * (-0.02)
    elif signal_to_noise_ratio < 20:
        noise_reduction = 0.70 + (signal_to_noise_ratio - 10) * (-0.015)
    elif signal_to_noise_ratio < 30:
        noise_reduction = 0.50 + (signal_to_noise_ratio - 20) * (-0.01)
    elif signal_to_noise_ratio < 40:
        noise_reduction = 0.35 + (signal_to_noise_ratio - 30) * (-0.0075)
    else:
        noise_reduction = 0.15
    
    noise_reduction = max(0.05, min(0.98, noise_reduction))
    
    if rms_coefficient_of_variation < 0.03 and dynamic_range < 5:
        noise_reduction = max(0.05, noise_reduction * 0.3)
    
    if silence_ratio_50db > 0.8:
        min_silence_duration = 0.25
    elif silence_ratio_50db > 0.6:
        min_silence_duration = 0.30
    elif silence_ratio_45db > 0.5:
        min_silence_duration = 0.35
    elif silence_ratio_40db > 0.4:
        min_silence_duration = 0.40
    else:
        min_silence_duration = 0.50
    
    if dynamic_range < 10:
        min_silence_duration = max(min_silence_duration, 0.4)
    
    target_db = -3.0
    
    has_low_freq_noise = noise_floor_db > -65
    highpass_cutoff = 100.0 if has_low_freq_noise else 0.0
    
    stationary_noise = signal_to_noise_ratio < 25 and rms_coefficient_of_variation < 0.2
    
    return {
        'noise_reduction': round(noise_reduction, 2),
        'silence_threshold': round(silence_threshold_db, 1),
        'min_silence_duration': round(min_silence_duration, 2),
        'target_db': round(target_db, 1),
        'stationary_noise': stationary_noise,
        'highpass_cutoff': round(highpass_cutoff, 0)
    }


def process_audio_adaptive(
    audio_data: np.ndarray,
    sample_rate: int,
    auto_detect: bool = True,
    noise_reduction: float = 0.0,
    silence_threshold: float = -40.0,
    min_silence_duration: float = 0.5,
    max_volume: bool = True,
    target_db: float = -1.0,
    stationary_noise: bool = False,
    scene: str = None,
    progress_callback=None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    def update_progress(pct, msg, status=None):
        if progress_callback:
            progress_callback(pct, msg, status)
    
    try:
        print(f"[Audio] 步骤1: 分析音频特征")
        analysis = analyze_audio_characteristics(audio_data, sample_rate)
        print(f"[Audio] 分析完成")
        
        if auto_detect:
            print(f"[Audio] 步骤2: 计算自适应参数")
            adaptive_params = calculate_adaptive_parameters(analysis)
            noise_reduction = adaptive_params['noise_reduction']
            silence_threshold = adaptive_params['silence_threshold']
            min_silence_duration = adaptive_params['min_silence_duration']
            target_db = adaptive_params['target_db']
            highpass_cutoff = adaptive_params['highpass_cutoff']
            print(f"[Audio] 参数: nr={noise_reduction}, hp={highpass_cutoff}, st={silence_threshold}")
        else:
            highpass_cutoff = 100.0
        
        if highpass_cutoff > 0:
            print(f"[Audio] 步骤3: 应用高通滤波 ({highpass_cutoff}Hz)")
            update_progress(0.05, '正在应用高通滤波...')
            audio_data = apply_highpass_filter(audio_data, sample_rate, highpass_cutoff)
            print(f"[Audio] 高通滤波完成")
        else:
            print(f"[Audio] 跳过高通滤波")
        
        if noise_reduction > 0:
            print(f"[Audio] 步骤4: 降噪处理 (强度: {noise_reduction})")
            update_progress(0.1, '正在进行降噪处理...')
            try:
                import noisereduce as nr
                
                chunk_duration = 30
                chunk_size = int(sample_rate * chunk_duration)
                total_samples = len(audio_data)
                
                if total_samples > chunk_size:
                    print(f"[Audio] 分块降噪: {total_samples//chunk_size + 1} 块")
                    reduced_chunks = []
                    total_chunks = (total_samples + chunk_size - 1) // chunk_size
                    for i, start in enumerate(range(0, total_samples, chunk_size)):
                        end = min(start + chunk_size + int(sample_rate * 1), total_samples)
                        chunk = audio_data[start:end]
                        
                        try:
                            reduced_chunk = nr.reduce_noise(
                                y=chunk,
                                sr=sample_rate,
                                prop_decrease=noise_reduction
                            )
                        except Exception as e:
                            print(f"[Audio] 降噪块 {i} 失败: {str(e)}")
                            reduced_chunk = chunk
                        
                        if i > 0:
                            reduced_chunk = reduced_chunk[int(sample_rate * 1):]
                        
                        reduced_chunks.append(reduced_chunk)
                        del chunk
                        
                        chunk_pct = (i + 1) / total_chunks
                        update_progress(0.1 + chunk_pct * 0.5, f'正在降噪处理: {int(chunk_pct * 100)}%', 'processing')
                    
                    import gc
                    gc.collect()
                    
                    reduced_noise = np.concatenate(reduced_chunks)
                    del reduced_chunks
                    gc.collect()
                else:
                    print(f"[Audio] 单块降噪")
                    reduced_noise = nr.reduce_noise(
                        y=audio_data,
                        sr=sample_rate,
                        prop_decrease=noise_reduction
                    )
                
                audio_data = audio_data * (1 - noise_reduction) + reduced_noise * noise_reduction
                del reduced_noise
                gc.collect()
                print(f"[Audio] 降噪完成")
            except Exception as e:
                print(f"[Audio] 降噪整体失败: {str(e)}，跳过")
        else:
            print(f"[Audio] 跳过高噪")
        
        print(f"[Audio] 步骤5: 静音检测")
        update_progress(0.65, '正在检测静音片段...', 'processing')
        
        frame_length = int(sample_rate * 0.02)
        hop_length = int(sample_rate * 0.01)
        
        detection_chunk_duration = 300
        detection_chunk_size = int(sample_rate * detection_chunk_duration)
        total_samples = len(audio_data)
        
        audio_normalized = audio_data / np.max(np.abs(audio_data)) if np.max(np.abs(audio_data)) > 0 else audio_data
        
        if total_samples > detection_chunk_size:
            silence_segments = []
            total_chunks = (total_samples + detection_chunk_size - 1) // detection_chunk_size
            
            for i, start in enumerate(range(0, total_samples, detection_chunk_size)):
                chunk = audio_normalized[start:start + detection_chunk_size]
                
                rms = librosa.feature.rms(y=chunk, frame_length=frame_length, hop_length=hop_length)
                db = librosa.amplitude_to_db(rms, ref=np.max)
                
                chunk_silence_mask = db < silence_threshold
                
                chunk_segments = []
                in_silence = False
                start_frame = 0
                
                for j, is_silent in enumerate(chunk_silence_mask[0]):
                    if is_silent and not in_silence:
                        in_silence = True
                        start_frame = j
                    elif not is_silent and in_silence:
                        in_silence = False
                        end_frame = j
                        duration = (end_frame - start_frame) * hop_length / sample_rate
                        if duration >= min_silence_duration:
                            start_time = (start + start_frame * hop_length) / sample_rate
                            end_time = (start + end_frame * hop_length) / sample_rate
                            chunk_segments.append((start_time, end_time))
                
                if in_silence:
                    end_frame = len(chunk_silence_mask[0])
                    duration = (end_frame - start_frame) * hop_length / sample_rate
                    if duration >= min_silence_duration:
                        start_time = (start + start_frame * hop_length) / sample_rate
                        end_time = (start + end_frame * hop_length) / sample_rate
                        chunk_segments.append((start_time, end_time))
                
                silence_segments.extend(chunk_segments)
                del chunk, rms, db, chunk_silence_mask, chunk_segments
                
                chunk_pct = (i + 1) / total_chunks
                update_progress(0.65 + chunk_pct * 0.1, f'正在检测静音: {int(chunk_pct * 100)}%', 'processing')
            
            import gc
            gc.collect()
        else:
            rms = librosa.feature.rms(y=audio_normalized, frame_length=frame_length, hop_length=hop_length)
            db = librosa.amplitude_to_db(rms, ref=np.max)
            
            silence_mask = db < silence_threshold
            
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
        
        if silence_segments:
            update_progress(0.75, '正在移除静音片段（保留50ms软边界）...', 'processing')
            
            segments_to_keep = []
            last_end = 0
            
            soft_boundary_ms = 50
            soft_boundary_samples = int(sample_rate * soft_boundary_ms / 1000)
            
            for start, end in sorted(silence_segments):
                if start > last_end:
                    segments_to_keep.append((last_end, start))
                last_end = end
            
            if last_end < len(audio_data) / sample_rate:
                segments_to_keep.append((last_end, len(audio_data) / sample_rate))
            
            if segments_to_keep:
                result_segments = []
                for idx, (start, end) in enumerate(segments_to_keep):
                    start_sample = int(start * sample_rate)
                    end_sample = int(end * sample_rate)
                    
                    segment = audio_data[start_sample:end_sample]
                    
                    if len(segment) > soft_boundary_samples * 2:
                        fade_in = np.linspace(0, 1, soft_boundary_samples)
                        fade_out = np.linspace(1, 0, soft_boundary_samples)
                        
                        segment[:soft_boundary_samples] = segment[:soft_boundary_samples] * fade_in
                        segment[-soft_boundary_samples:] = segment[-soft_boundary_samples:] * fade_out
                    
                    result_segments.append(segment)
                
                audio_data = np.concatenate(result_segments)
        
        update_progress(0.9, '正在调整音量...', 'processing')
        
        if max_volume:
            rms = np.sqrt(np.mean(audio_data**2))
            if rms > 0:
                current_db = 20 * np.log10(rms)
                gain = 10 ** ((target_db - current_db) / 20)
                audio_data = audio_data * gain
                
                max_val = np.max(np.abs(audio_data))
                if max_val > 1.0:
                    audio_data = audio_data / max_val
        
        original_duration = len(audio_data) / sample_rate
        
        return audio_data, {
            'success': True,
            'analysis': analysis,
            'applied_params': {
                'noise_reduction': noise_reduction,
                'silence_threshold': silence_threshold,
                'min_silence_duration': min_silence_duration,
                'target_db': target_db,
                'stationary_noise': stationary_noise,
                'highpass_cutoff': highpass_cutoff,
                'auto_detect': auto_detect,
                'scene': scene
            },
            'stats': {
                'duration': round(original_duration, 2),
                'sample_rate': sample_rate,
                'silence_segments_removed': len(silence_segments)
            }
        }
    
    except ImportError:
        return audio_data, {
            'success': False,
            'analysis': analysis,
            'message': 'noisereduce 模块未安装，跳过降噪'
        }
    except Exception as e:
        return audio_data, {
            'success': False,
            'analysis': analysis,
            'message': f'自适应处理失败: {str(e)}'
        }
