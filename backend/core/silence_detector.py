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
    chunk_size = int(sample_rate * chunk_duration)
    overlap_size = int(sample_rate * 1.0)

    all_silence_segments = []
    total_chunks = (total_samples + chunk_size - 1) // chunk_size

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
            if i > 0 and seg_start < 1.0:
                continue
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
