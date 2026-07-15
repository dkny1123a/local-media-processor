import os
import tempfile

import librosa
import numpy as np
import soundfile as sf

from .core import (
    convert_to_wav,
    detect_silence,
    encode_to_mp3,
    remove_silence,
)


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
        success = encode_to_mp3(temp_path, output_path, sample_rate)
        return success
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
                verbose=False,
            )
        else:
            reduced_noise = nr.reduce_noise(
                y=audio_data,
                sr=sample_rate,
                verbose=False,
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


def process_audio(
    input_path,
    output_path,
    silence_threshold=-40.0,
    min_silence_duration=0.5,
    max_volume=True,
    noise_reduction=0.0,
    stationary_noise=False,
):
    try:
        output_path = os.path.splitext(output_path)[0] + '.mp3'

        converted_path, was_converted = convert_to_wav(input_path)

        audio_data, sample_rate = librosa.load(converted_path, sr=None, mono=True)

        if was_converted:
            os.unlink(converted_path)

        original_duration = len(audio_data) / sample_rate

        audio_data = apply_denoise(audio_data, sample_rate, noise_reduction, stationary_noise)

        silence_segments = detect_silence(
            audio_data,
            sample_rate,
            threshold_dbfs=silence_threshold,
            min_silence_duration=min_silence_duration,
        )

        processed_audio = remove_silence(audio_data, sample_rate, silence_segments)

        if max_volume:
            processed_audio = normalize_volume(processed_audio)

        success = save_as_mp3(processed_audio, sample_rate, output_path)

        if not success:
            return {
                "success": False,
                "message": "音频处理失败: 无法保存为MP3格式",
            }

        processed_duration = len(processed_audio) / sample_rate
        silence_removed = len(silence_segments)
        duration_reduction = (
            ((original_duration - processed_duration) / original_duration * 100)
            if original_duration > 0
            else 0
        )

        return {
            "success": True,
            "message": "音频处理完成",
            "stats": {
                "original_duration": round(original_duration, 2),
                "processed_duration": round(processed_duration, 2),
                "silence_segments_removed": silence_removed,
                "duration_reduction_percent": round(duration_reduction, 2),
                "noise_reduction": noise_reduction,
                "sample_rate": sample_rate,
            },
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"音频处理失败: {str(e)}",
        }
