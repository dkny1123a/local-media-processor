from typing import List, Tuple

import numpy as np

from .constants import DEFAULT_SOFT_BOUNDARY_MS


def compute_segments_to_keep(
    total_duration: float,
    silence_segments: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    if not silence_segments:
        return [(0.0, total_duration)]

    segments_to_keep = []
    last_end = 0.0

    for start, end in sorted(silence_segments):
        if start > last_end:
            segments_to_keep.append((last_end, start))
        last_end = max(last_end, end)

    if last_end < total_duration:
        segments_to_keep.append((last_end, total_duration))

    return segments_to_keep


def apply_fade(
    segment: np.ndarray,
    sample_rate: int,
    soft_boundary_ms: int = DEFAULT_SOFT_BOUNDARY_MS,
) -> np.ndarray:
    if len(segment) == 0:
        return segment

    soft_boundary_samples = int(sample_rate * soft_boundary_ms / 1000)

    if len(segment) <= soft_boundary_samples * 2:
        return segment

    result = segment.copy()

    fade_in = np.linspace(0, 1, soft_boundary_samples)
    fade_out = np.linspace(1, 0, soft_boundary_samples)

    result[:soft_boundary_samples] = result[:soft_boundary_samples] * fade_in
    result[-soft_boundary_samples:] = result[-soft_boundary_samples:] * fade_out

    return result


def remove_silence(
    audio_data: np.ndarray,
    sample_rate: int,
    silence_segments: List[Tuple[float, float]],
    soft_boundary_ms: int = DEFAULT_SOFT_BOUNDARY_MS,
) -> np.ndarray:
    if not silence_segments or len(audio_data) == 0:
        return audio_data

    total_duration = len(audio_data) / sample_rate
    segments_to_keep = compute_segments_to_keep(total_duration, silence_segments)

    if not segments_to_keep:
        return audio_data

    result_segments = []

    for start, end in segments_to_keep:
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)

        segment = audio_data[start_sample:end_sample]

        if len(segment) > 0:
            segment = apply_fade(segment, sample_rate, soft_boundary_ms)
            result_segments.append(segment)

    if not result_segments:
        return audio_data

    return np.concatenate(result_segments)


def remove_silence_from_chunk(
    audio_chunk: np.ndarray,
    sample_rate: int,
    silence_segments: List[Tuple[float, float]],
    soft_boundary_ms: int = DEFAULT_SOFT_BOUNDARY_MS,
) -> np.ndarray:
    return remove_silence(audio_chunk, sample_rate, silence_segments, soft_boundary_ms)
