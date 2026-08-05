import numpy as np
import librosa
from typing import Dict, Any



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
    from .core.audio_analyzer import analyze_audio_spectrum
    base = analyze_audio_spectrum(audio_data, sample_rate)
    
    frame_length = int(sample_rate * 0.02)
    hop_length = int(sample_rate * 0.01)
    
    total_samples = len(audio_data)
    analysis_duration = 60
    analysis_samples = int(sample_rate * analysis_duration)
    
    all_rms = []
    num_segments = min(10, max(1, total_samples // analysis_samples))
    
    for i in range(num_segments):
        offset = int(i * (total_samples - analysis_samples) / max(num_segments - 1, 1)) if num_segments > 1 else 0
        segment = audio_data[offset:offset + analysis_samples]
        
        if len(segment) == 0:
            continue
        
        rms = librosa.feature.rms(y=segment, frame_length=frame_length, hop_length=hop_length)[0]
        all_rms.extend(rms)
    
    if len(all_rms) == 0:
        rms_db = np.array([-80.0])
    else:
        rms_db = librosa.amplitude_to_db(np.array(all_rms), ref=1.0)
    
    silence_ratio_50db = float(np.mean(rms_db < -50))
    silence_ratio_45db = float(np.mean(rms_db < -45))
    silence_ratio_40db = float(np.mean(rms_db < -40))
    
    rms_array = np.array(all_rms) if all_rms else np.array([0])
    rms_mean = float(np.mean(rms_array))
    rms_std = float(np.std(rms_array))
    
    return {
        **base,
        'rms_mean': rms_mean,
        'rms_median': float(np.median(rms_array)),
        'rms_std': rms_std,
        'rms_min': float(np.min(rms_array)),
        'rms_max': float(np.max(rms_array)),
        'signal_to_noise_ratio': base['dynamic_range'],
        'rms_coefficient_of_variation': rms_std / rms_mean if rms_mean > 0 else 0.0,
        'silence_ratio_50db': silence_ratio_50db,
        'silence_ratio_45db': silence_ratio_45db,
        'silence_ratio_40db': silence_ratio_40db,
        'avg_quiet_frame_db': float(np.mean(rms_db[rms_db < np.percentile(rms_db, 20)])) if len(rms_db) > 0 else base['noise_floor_db'],
        'avg_loud_frame_db': float(np.mean(rms_db[rms_db > np.percentile(rms_db, 80)])) if len(rms_db) > 0 else base['signal_peak_db']
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

    # ============================================================
    # 自适应静音阈值：基于"人耳不可辨识"原理
    # ============================================================
    # 人耳不可辨识判定：信号被环境噪声完全掩盖
    #   - 信号能量 ≤ noise_floor + 3dB 时，被噪声掩盖听不到
    #   - 阈值跟随音频自身噪声底噪变化，不硬编码绝对值
    #
    # 核心公式：noise_floor + 3dB（同时掩蔽临界值）
    # 范围约束：[noise_floor + 1dB, noise_floor + 10dB]
    #   - 不再依赖 signal_peak（当 signal_peak 接近 noise_floor 时
    #     旧的 signal_peak - 15dB 边界会把阈值压到过低水平）
    # ============================================================
    silence_threshold_db = noise_floor_db + 3.0
    silence_threshold_db = min(silence_threshold_db, noise_floor_db + 10.0)
    silence_threshold_db = max(silence_threshold_db, noise_floor_db + 1.0)

    if signal_to_noise_ratio < 5:
        noise_reduction = 0.95
    elif signal_to_noise_ratio < 10:
        noise_reduction = 0.90 + (signal_to_noise_ratio - 5) * (-0.01)
    elif signal_to_noise_ratio < 20:
        noise_reduction = 0.82 + (signal_to_noise_ratio - 10) * (-0.008)
    elif signal_to_noise_ratio < 30:
        noise_reduction = 0.72 + (signal_to_noise_ratio - 20) * (-0.006)
    elif signal_to_noise_ratio < 40:
        noise_reduction = 0.60 + (signal_to_noise_ratio - 30) * (-0.005)
    else:
        noise_reduction = 0.45

    noise_reduction = max(0.55, min(0.98, noise_reduction))

    if rms_coefficient_of_variation < 0.03 and dynamic_range < 5:
        noise_reduction = max(0.05, noise_reduction * 0.3)

    # 最小静音时长：0.8s（人耳不可辨识的持续时长门槛）
    # ============================================================
    # 原 1.5s 过于保守，会保留大量无信息停顿（语句间停顿、呼吸换气等）
    # 0.8s 选择依据：
    #   - 正常说话换气停顿 < 0.5s，不会被误移除
    #   - 0.8s+ 的被噪声完全掩盖片段确实无信息量
    #   - 介于"避免误删换气"和"有效压缩无信息段"之间
    # ============================================================
    min_silence_duration = 0.8

    # 目标响度 -14 LUFS（骑行嘈杂环境，非流媒体-16）
    # 第一性原理：收听端骑行噪声~75dB SPL，需更高响度才能听清
    # compand+volume 方案不直接使用此值，volume=25dB 固定增益
    target_db = -14.0

    has_low_freq_noise = noise_floor_db > -65
    highpass_cutoff = 150.0 if has_low_freq_noise else 0.0

    stationary_noise = signal_to_noise_ratio < 25 and rms_coefficient_of_variation < 0.2

    print(f"[Adaptive] noise_floor={noise_floor_db:.1f}dB, signal_peak={signal_peak_db:.1f}dB, "
          f"dynamic_range={dynamic_range:.1f}dB → silence_threshold={silence_threshold_db:.1f}dB "
          f"(noise_floor+{silence_threshold_db - noise_floor_db:.1f}dB), min_silence={min_silence_duration}s")

    return {
        'noise_reduction': round(noise_reduction, 2),
        'silence_threshold': round(silence_threshold_db, 1),
        'min_silence_duration': round(min_silence_duration, 2),
        'target_db': round(target_db, 1),
        'stationary_noise': stationary_noise,
        'highpass_cutoff': round(highpass_cutoff, 0)
    }
