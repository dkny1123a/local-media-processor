from .constants import (
    SUPPORTED_AUDIO_EXTS,
    SUPPORTED_VIDEO_EXTS,
    SUPPORTED_EXTS,
    AUDIO_MIME_TYPES,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_CHANNELS,
    DEFAULT_FRAME_LENGTH_SEC,
    DEFAULT_HOP_LENGTH_SEC,
    DEFAULT_SOFT_BOUNDARY_MS,
    DEFAULT_SILENCE_THRESHOLD_DBFS,
    DEFAULT_MIN_SILENCE_DURATION_SEC,
)

from .ffmpeg_utils import (
    convert_to_wav,
    get_audio_duration,
    load_audio_chunk,
    load_audio_chunk_ffmpeg,
    load_audio_chunk_librosa,
    encode_to_mp3,
    resample_audio,
    apply_loudnorm,
    concatenate_audio_files,
)

from .silence_detector import (
    detect_silence,
    detect_silence_chunked,
    compute_silence_ratio,
)

from .silence_remover import (
    remove_silence,
    remove_silence_from_chunk,
    compute_segments_to_keep,
    apply_fade,
)

__all__ = [
    'SUPPORTED_AUDIO_EXTS',
    'SUPPORTED_VIDEO_EXTS',
    'SUPPORTED_EXTS',
    'AUDIO_MIME_TYPES',
    'DEFAULT_SAMPLE_RATE',
    'DEFAULT_CHANNELS',
    'DEFAULT_FRAME_LENGTH_SEC',
    'DEFAULT_HOP_LENGTH_SEC',
    'DEFAULT_SOFT_BOUNDARY_MS',
    'DEFAULT_SILENCE_THRESHOLD_DBFS',
    'DEFAULT_MIN_SILENCE_DURATION_SEC',
    'convert_to_wav',
    'get_audio_duration',
    'load_audio_chunk',
    'load_audio_chunk_ffmpeg',
    'load_audio_chunk_librosa',
    'encode_to_mp3',
    'resample_audio',
    'apply_loudnorm',
    'concatenate_audio_files',
    'detect_silence',
    'detect_silence_chunked',
    'compute_silence_ratio',
    'remove_silence',
    'remove_silence_from_chunk',
    'compute_segments_to_keep',
    'apply_fade',
]
