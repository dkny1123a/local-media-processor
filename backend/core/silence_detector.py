from typing import List, Optional, Tuple

import numpy as np

from .constants import (
    DEFAULT_FRAME_LENGTH_SEC,
    DEFAULT_HOP_LENGTH_SEC,
    DEFAULT_MIN_SILENCE_DURATION_SEC,
    DEFAULT_SILENCE_THRESHOLD_DBFS,
)


def _compute_rms_db(
    audio_data: np.ndarray,
    sample_rate: int,
    frame_length: float = DEFAULT_FRAME_LENGTH_SEC,
    hop_length: float = DEFAULT_HOP_LENGTH_SEC,
) -> np.ndarray:
    import librosa

    n_fft = int(sample_rate * frame_length)
    hop_len = int(sample_rate * hop_length)

    rms = librosa.feature.rms(y=audio_data, frame_length=n_fft, hop_length=hop_len)[0]
    db = librosa.amplitude_to_db(rms, ref=1.0)

    return db


def _find_silence_segments(
    db: np.ndarray,
    sample_rate: int,
    threshold_dbfs: float,
    min_silence_duration: float,
    hop_length: float = DEFAULT_HOP_LENGTH_SEC,
    offset_time: float = 0.0,
) -> List[Tuple[float, float]]:
    silence_mask = db < threshold_dbfs

    silence_segments = []
    in_silence = False
    start_frame = 0

    for i, is_silent in enumerate(silence_mask):
        if is_silent and not in_silence:
            in_silence = True
            start_frame = i
        elif not is_silent and in_silence:
            in_silence = False
            end_frame = i
            duration = (end_frame - start_frame) * hop_length
            if duration >= min_silence_duration:
                start_time = offset_time + start_frame * hop_length
                end_time = offset_time + end_frame * hop_length
                silence_segments.append((start_time, end_time))

    if in_silence:
        end_frame = len(silence_mask)
        duration = (end_frame - start_frame) * hop_length
        if duration >= min_silence_duration:
            start_time = offset_time + start_frame * hop_length
            end_time = offset_time + len(silence_mask) * hop_length
            silence_segments.append((start_time, end_time))

    return silence_segments


def detect_silence(
    audio_data: np.ndarray,
    sample_rate: int,
    threshold_dbfs: float = DEFAULT_SILENCE_THRESHOLD_DBFS,
    min_silence_duration: float = DEFAULT_MIN_SILENCE_DURATION_SEC,
    frame_length: float = DEFAULT_FRAME_LENGTH_SEC,
    hop_length: float = DEFAULT_HOP_LENGTH_SEC,
) -> List[Tuple[float, float]]:
    db = _compute_rms_db(audio_data, sample_rate, frame_length, hop_length)
    return _find_silence_segments(db, sample_rate, threshold_dbfs, min_silence_duration, hop_length)


def detect_silence_chunked(
    audio_data: np.ndarray,
    sample_rate: int,
    threshold_dbfs: float = DEFAULT_SILENCE_THRESHOLD_DBFS,
    min_silence_duration: float = DEFAULT_MIN_SILENCE_DURATION_SEC,
    frame_length: float = DEFAULT_FRAME_LENGTH_SEC,
    hop_length: float = DEFAULT_HOP_LENGTH_SEC,
    chunk_duration: float = 300,
) -> List[Tuple[float, float]]:
    total_samples = len(audio_data)

    # 边界条件：空音频或极短音频直接返回空列表
    if total_samples == 0:
        return []

    chunk_size = int(sample_rate * chunk_duration)
    overlap_size = int(sample_rate * 1.0)

    all_silence_segments = []
    total_chunks = (total_samples + chunk_size - 1) // chunk_size

    # 边界条件：音频短于一个 chunk，直接单次检测
    if total_chunks <= 1:
        return detect_silence(
            audio_data, sample_rate,
            threshold_dbfs=threshold_dbfs,
            min_silence_duration=min_silence_duration,
            frame_length=frame_length,
            hop_length=hop_length,
        )

    for i in range(total_chunks):
        start = i * chunk_size
        end = min(start + chunk_size + overlap_size, total_samples)
        chunk = audio_data[start:end]
        offset_time = start / sample_rate

        chunk_segments = detect_silence(
            chunk, sample_rate,
            threshold_dbfs=threshold_dbfs,
            min_silence_duration=min_silence_duration,
            frame_length=frame_length,
            hop_length=hop_length,
        )

        adjusted_segments = []
        for seg_start, seg_end in chunk_segments:
            # 过滤掉落在 overlap 区域的静音段（避免重复检测）
            # 但保留首 chunk 的所有段（i==0 时不过滤）
            if i > 0 and seg_start < 1.0:
                continue
            # 过滤掉延伸到下一个 chunk overlap 区域的段尾部
            # （会在下一个 chunk 中被完整检测到）
            adjusted_segments.append((offset_time + seg_start, offset_time + seg_end))

        all_silence_segments.extend(adjusted_segments)

    return _merge_silence_segments(all_silence_segments, min_silence_duration)


def _merge_silence_segments(
    segments: List[Tuple[float, float]],
    min_silence_duration: float,
) -> List[Tuple[float, float]]:
    if not segments:
        return []

    sorted_segments = sorted(segments, key=lambda x: x[0])
    merged = [sorted_segments[0]]

    for current in sorted_segments[1:]:
        last = merged[-1]
        if current[0] <= last[1] + 0.01:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)

    return [(s, e) for s, e in merged if (e - s) >= min_silence_duration]


def compute_silence_ratio(
    audio_data: np.ndarray,
    sample_rate: int,
    threshold_dbfs: float = DEFAULT_SILENCE_THRESHOLD_DBFS,
) -> float:
    db = _compute_rms_db(audio_data, sample_rate)
    return float(np.mean(db < threshold_dbfs))


def remove_silence_segments(
    audio_data: np.ndarray,
    sample_rate: int,
    silence_segments: List[Tuple[float, float]],
    fade_ms: int = 5,
) -> np.ndarray:
    """Remove silent segments from audio data with fade in/out to avoid clicks.

    问题3修复：原实现直接 concatenate，拼接点会有 DC offset 跳变 → 爆音
    修复：对每个保留段首尾加 fade_ms 毫秒的线性淡入淡出

    Args:
        audio_data: Input audio numpy array
        sample_rate: Audio sample rate
        silence_segments: List of (start_time, end_time) tuples in seconds
        fade_ms: Fade in/out duration in milliseconds (default 5ms)

    Returns:
        Audio data with silent segments removed
    """
    if not silence_segments:
        return audio_data

    sorted_segments = sorted(silence_segments, key=lambda x: x[0])

    keep_ranges = []
    last_end = 0.0
    for start, end in sorted_segments:
        if start > last_end:
            keep_ranges.append((last_end, start))
        last_end = max(last_end, end)

    if last_end < len(audio_data) / sample_rate:
        keep_ranges.append((last_end, len(audio_data) / sample_rate))

    if not keep_ranges:
        # 边界条件：全静音音频，返回极小静音段避免下游写空数组崩溃
        # 至少保留 1ms 的静音，让 sf.write 能正常写入
        min_samples = max(1, int(sample_rate * 0.001))
        return np.zeros(min_samples, dtype=audio_data.dtype)

    fade_samples = int(sample_rate * fade_ms / 1000)
    # 边界条件：fade_samples 为 0 时（极低采样率），跳过淡入淡出
    if fade_samples < 1:
        fade_samples = 1

    chunks = []
    for start, end in keep_ranges:
        start_idx = int(start * sample_rate)
        end_idx = int(end * sample_rate)
        if start_idx < len(audio_data) and end_idx <= len(audio_data) and start_idx < end_idx:
            segment = audio_data[start_idx:end_idx].copy()

            # 仅对足够长的段应用淡入淡出，避免段过短时被衰减殆尽
            if len(segment) > fade_samples * 2:
                fade_in = np.linspace(0, 1, fade_samples)
                fade_out = np.linspace(1, 0, fade_samples)
                segment[:fade_samples] *= fade_in
                segment[-fade_samples:] *= fade_out

            chunks.append(segment)

    if not chunks:
        # 所有 keep_ranges 都无效（理论上不该发生），返回极小静音段
        min_samples = max(1, int(sample_rate * 0.001))
        return np.zeros(min_samples, dtype=audio_data.dtype)

    return np.concatenate(chunks)
