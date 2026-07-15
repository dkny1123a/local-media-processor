from typing import Dict, List

SUPPORTED_AUDIO_EXTS: List[str] = ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.amr', '.3gp']
SUPPORTED_VIDEO_EXTS: List[str] = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
SUPPORTED_EXTS: List[str] = SUPPORTED_AUDIO_EXTS + SUPPORTED_VIDEO_EXTS

AUDIO_MIME_TYPES: Dict[str, str] = {
    '.mp3': 'audio/mpeg',
    '.wav': 'audio/wav',
    '.flac': 'audio/flac',
    '.ogg': 'audio/ogg',
    '.aac': 'audio/aac',
    '.m4a': 'audio/mp4',
    '.amr': 'audio/amr',
    '.3gp': 'audio/3gpp',
}

AMR_FORMATS: List[str] = ['.amr', '.3gp']
M4A_FORMATS: List[str] = ['.m4a']
FORMATS_REQUIRING_CONVERSION: List[str] = AMR_FORMATS + M4A_FORMATS

DEFAULT_FRAME_LENGTH_SEC: float = 0.02
DEFAULT_HOP_LENGTH_SEC: float = 0.01
DEFAULT_SOFT_BOUNDARY_MS: int = 50

DEFAULT_SILENCE_THRESHOLD_DBFS: float = -40.0
DEFAULT_MIN_SILENCE_DURATION_SEC: float = 0.5
DEFAULT_CHUNK_DURATION_SEC: int = 60

FFMPEG_DEFAULT_TIMEOUT: int = 120
FFPROBE_DEFAULT_TIMEOUT: int = 30
FFMPEG_CHUNK_TIMEOUT: int = 60
FFMPEG_ENCODE_TIMEOUT: int = 600

DEFAULT_SAMPLE_RATE: int = 44100
DEFAULT_CHANNELS: int = 1
DEFAULT_MP3_QUALITY: str = '2'
