import os
import subprocess
import tempfile
from typing import Optional, Tuple

import numpy as np

from .constants import (
    DEFAULT_CHANNELS,
    DEFAULT_MP3_QUALITY,
    DEFAULT_SAMPLE_RATE,
    FFMPEG_CHUNK_TIMEOUT,
    FFMPEG_DEFAULT_TIMEOUT,
    FFMPEG_ENCODE_TIMEOUT,
    FFPROBE_DEFAULT_TIMEOUT,
    FORMATS_REQUIRING_CONVERSION,
)


def convert_to_wav(
    input_path: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    timeout: int = FFMPEG_DEFAULT_TIMEOUT,
) -> Tuple[str, bool]:
    ext = os.path.splitext(input_path)[1].lower()

    if ext not in FORMATS_REQUIRING_CONVERSION:
        return input_path, False

    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_path = temp_wav.name
    temp_wav.close()

    command = [
        'ffmpeg',
        '-i', input_path,
        '-ac', str(channels),
        '-ar', str(sample_rate),
        '-y',
        temp_path,
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, timeout=timeout)
        return temp_path, True
    except subprocess.CalledProcessError as e:
        print(f"格式转换失败: {e.stderr.decode()}")
        return input_path, False
    except subprocess.TimeoutExpired:
        print(f"格式转换超时")
        os.unlink(temp_path)
        return input_path, False


def get_audio_duration(
    file_path: str,
    timeout: int = FFPROBE_DEFAULT_TIMEOUT,
) -> float:
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
             '-of', 'csv=p=0', file_path],
            capture_output=True, text=True, check=True, timeout=timeout,
        )
        return float(result.stdout.strip())
    except subprocess.TimeoutExpired:
        print(f"ffprobe超时: {file_path}")
    except Exception:
        pass

    try:
        import librosa
        return librosa.get_duration(path=file_path)
    except Exception:
        return 0.0


def load_audio_chunk_ffmpeg(
    file_path: str,
    sample_rate: int,
    offset: float,
    duration: float,
    timeout: int = FFMPEG_CHUNK_TIMEOUT,
) -> Optional[np.ndarray]:
    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_path = temp_wav.name
    temp_wav.close()

    command = [
        'ffmpeg',
        '-ss', str(offset),
        '-i', file_path,
        '-t', str(duration),
        '-ac', '1',
        '-ar', str(sample_rate),
        '-f', 'wav',
        '-y',
        '-loglevel', 'quiet',
        temp_path,
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, timeout=timeout)

        import librosa
        chunk, _ = librosa.load(temp_path, sr=sample_rate, mono=True)

        os.unlink(temp_path)
        return chunk
    except subprocess.TimeoutExpired:
        os.unlink(temp_path)
        return None
    except Exception:
        os.unlink(temp_path)
        return None


def load_audio_chunk_librosa(
    file_path: str,
    sample_rate: int,
    offset: float,
    duration: float,
) -> Optional[np.ndarray]:
    try:
        import librosa
        chunk = librosa.load(
            file_path, sr=sample_rate, mono=True,
            offset=offset, duration=duration,
        )[0]
        return chunk
    except Exception:
        return None


def load_audio_chunk(
    file_path: str,
    sample_rate: int,
    offset: float,
    duration: float,
) -> Optional[np.ndarray]:
    chunk = load_audio_chunk_ffmpeg(file_path, sample_rate, offset, duration)
    if chunk is None:
        chunk = load_audio_chunk_librosa(file_path, sample_rate, offset, duration)
    return chunk


def encode_to_mp3(
    input_path: str,
    output_path: str,
    sample_rate: int,
    channels: int = DEFAULT_CHANNELS,
    quality: str = DEFAULT_MP3_QUALITY,
    timeout: int = FFMPEG_ENCODE_TIMEOUT,
) -> bool:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    command = [
        'ffmpeg',
        '-i', input_path,
        '-ac', str(channels),
        '-ar', str(sample_rate),
        '-c:a', 'libmp3lame',
        '-q:a', quality,
        '-y',
        '-loglevel', 'quiet',
        output_path,
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        print(f"MP3编码超时: {output_path}")
        return False
    except subprocess.CalledProcessError as e:
        print(f"MP3编码失败: {e.stderr.decode()}")
        return False


def resample_audio(
    input_path: str,
    output_path: str,
    target_sample_rate: int,
    channels: int = DEFAULT_CHANNELS,
    timeout: int = FFMPEG_DEFAULT_TIMEOUT,
) -> bool:
    command = [
        'ffmpeg',
        '-i', input_path,
        '-ac', str(channels),
        '-ar', str(target_sample_rate),
        '-y',
        '-loglevel', 'quiet',
        output_path,
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        print(f"重采样超时: {input_path}")
        return False
    except subprocess.CalledProcessError as e:
        print(f"重采样失败: {e.stderr.decode()}")
        return False


def apply_loudnorm(
    input_path: str,
    output_path: str,
    target_db: float = -1.0,
    timeout: int = FFMPEG_ENCODE_TIMEOUT,
) -> bool:
    command = [
        'ffmpeg',
        '-i', input_path,
        '-af', f'loudnorm=I={target_db}:LRA=11:TP=-1.5',
        '-y',
        '-loglevel', 'quiet',
        output_path,
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        print(f"音量调整超时: {input_path}")
        return False
    except subprocess.CalledProcessError as e:
        print(f"音量调整失败: {e.stderr.decode()}")
        return False


def concatenate_audio_files(
    input_files: list,
    output_path: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    quality: str = DEFAULT_MP3_QUALITY,
    timeout: int = FFMPEG_ENCODE_TIMEOUT,
) -> bool:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    list_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    list_path = list_file.name
    try:
        for file_path in input_files:
            list_file.write(f"file '{file_path}'\n")
    finally:
        list_file.close()

    command = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', list_path,
        '-c', 'copy',
        '-y',
        '-loglevel', 'quiet',
        output_path,
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, timeout=timeout)
        os.unlink(list_path)
        return True
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        os.unlink(list_path)
        temp_wav_files = []
        try:
            for file_path in input_files:
                temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_wav_path = temp_wav.name
                temp_wav.close()

                convert_cmd = [
                    'ffmpeg',
                    '-i', file_path,
                    '-ac', str(channels),
                    '-ar', str(sample_rate),
                    '-y',
                    '-loglevel', 'quiet',
                    temp_wav_path,
                ]
                subprocess.run(convert_cmd, check=True, capture_output=True, timeout=FFMPEG_DEFAULT_TIMEOUT)
                temp_wav_files.append(temp_wav_path)

            concat_list_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
            concat_list_path = concat_list_file.name
            try:
                for wav_file in temp_wav_files:
                    concat_list_file.write(f"file '{wav_file}'\n")
            finally:
                concat_list_file.close()

            concat_cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_list_path,
                '-ac', str(channels),
                '-ar', str(sample_rate),
                '-c:a', 'libmp3lame',
                '-q:a', quality,
                '-y',
                '-loglevel', 'quiet',
                output_path,
            ]
            subprocess.run(concat_cmd, check=True, capture_output=True, timeout=timeout)
            os.unlink(concat_list_path)

            for wav_file in temp_wav_files:
                if os.path.exists(wav_file):
                    os.unlink(wav_file)
            return True
        except Exception as convert_e:
            for wav_file in temp_wav_files:
                if os.path.exists(wav_file):
                    os.unlink(wav_file)
            print(f"拼接失败: {str(convert_e)}")
            return False
