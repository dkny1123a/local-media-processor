import os
import subprocess
import tempfile
from typing import Any, Dict, Tuple

import librosa
import numpy as np

from .core import apply_fade, apply_loudnorm


def apply_preemphasis(audio_data: np.ndarray, coefficient: float = 0.97) -> np.ndarray:
    try:
        preemphasized = np.append(audio_data[0], audio_data[1:] - coefficient * audio_data[:-1])
        return preemphasized
    except Exception as e:
        print(f"预加重滤波器失败: {e}")
        return audio_data


def apply_deemphasis(audio_data: np.ndarray, coefficient: float = 0.97) -> np.ndarray:
    try:
        from scipy.signal import lfilter
        return lfilter([1.0], [1.0, -coefficient], audio_data)
    except Exception as e:
        print(f"去加重滤波器失败: {e}")
        return audio_data


def apply_dynamic_range_compression(
    audio_data: np.ndarray,
    sample_rate: int,
    ratio: float = 2.5,
    threshold_db: float = -25.0,
    knee_db: float = 6.0,
    makeup_gain_db: float = 0.0,
) -> np.ndarray:
    try:
        frame_length = int(sample_rate * 0.05)
        hop_length = int(sample_rate * 0.025)

        rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=1.0)

        gain_db = np.zeros_like(rms_db)

        above_knee = rms_db > threshold_db + knee_db
        in_knee = (rms_db > threshold_db - knee_db) & (~above_knee)

        gain_db[above_knee] = (threshold_db - rms_db[above_knee]) * (1 - 1.0 / ratio)

        knee_input = rms_db[in_knee] - threshold_db + knee_db
        gain_db[in_knee] = (1 - 1.0 / ratio) * knee_input ** 2 / (4 * knee_db)

        gain_db += makeup_gain_db

        gain = librosa.db_to_amplitude(gain_db)

        gain_expanded = np.repeat(gain, hop_length)[:len(audio_data)]
        if len(gain_expanded) < len(audio_data):
            gain_expanded = np.pad(gain_expanded, (0, len(audio_data) - len(gain_expanded)), 'edge')

        compressed_audio = audio_data * gain_expanded

        return compressed_audio
    except Exception as e:
        print(f"动态范围压缩失败: {e}")
        return audio_data


def apply_bandpass_filter(
    audio_data: np.ndarray,
    sample_rate: int,
    low_cut: float = 80.0,
    high_cut: float = 8000.0,
) -> np.ndarray:
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
            temp_output_path,
        ]

        subprocess.run(command, check=True, capture_output=True, timeout=120)

        filtered_audio, _ = librosa.load(temp_output_path, sr=sample_rate, mono=True)

        os.unlink(temp_input_path)
        os.unlink(temp_output_path)

        return filtered_audio
    except subprocess.TimeoutExpired:
        os.unlink(temp_input_path)
        os.unlink(temp_output_path)
        print("带通滤波超时")
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
        n_fft = 512
        hop_length = 256

        stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
        magnitude, phase = librosa.magphase(stft)

        freq_bins = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)

        voice_mask = np.ones(len(freq_bins))
        for i, freq in enumerate(freq_bins):
            if 300 <= freq <= 2000:
                voice_mask[i] = 1.10
            elif 2000 < freq <= 4000:
                voice_mask[i] = 1.20
            elif 4000 < freq <= 6000:
                voice_mask[i] = 1.08
            elif 80 <= freq < 300:
                voice_mask[i] = 0.85
            elif 6000 < freq <= 8000:
                voice_mask[i] = 0.70

        enhanced_magnitude = magnitude * voice_mask[:, np.newaxis]

        enhanced_stft = enhanced_magnitude * phase
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=hop_length)

        if len(enhanced_audio) > len(audio_data):
            enhanced_audio = enhanced_audio[:len(audio_data)]
        elif len(enhanced_audio) < len(audio_data):
            padding = np.zeros(len(audio_data) - len(enhanced_audio))
            enhanced_audio = np.concatenate([enhanced_audio, padding])

        return enhanced_audio
    except Exception as e:
        print(f"语音增强失败: {e}")
        return audio_data


def detect_voice_activity(
    audio_data: np.ndarray,
    sample_rate: int,
    threshold_db: float = -45.0,
    min_duration: float = 0.1,
) -> np.ndarray:
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
                    voice_segments.append(
                        (start_frame * hop_length / sample_rate, end_frame * hop_length / sample_rate)
                    )

        if in_voice:
            end_frame = len(voice_mask)
            duration = (end_frame - start_frame) * hop_length / sample_rate
            if duration >= min_duration:
                voice_segments.append(
                    (start_frame * hop_length / sample_rate, len(audio_data) / sample_rate)
                )

        return voice_segments
    except Exception as e:
        print(f"语音活动检测失败: {e}")
        return [(0.0, len(audio_data) / sample_rate)]


def apply_intelligibility_boost(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    try:
        n_fft = 512
        hop_length = int(sample_rate * 0.01)

        stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
        magnitude, phase = librosa.magphase(stft)

        freq_bins = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
        num_frames = magnitude.shape[1]

        frame_length_rms = int(sample_rate * 0.02)
        rms = librosa.feature.rms(y=audio_data, frame_length=frame_length_rms, hop_length=hop_length)[0]
        db = librosa.amplitude_to_db(rms, ref=1.0)

        if len(db) != num_frames:
            db = np.interp(
                np.linspace(0, len(db) - 1, num_frames),
                np.arange(len(db)),
                db
            )

        freq_gain = np.ones(len(freq_bins))
        voice_band = np.zeros(len(freq_bins), dtype=bool)
        for i, freq in enumerate(freq_bins):
            if 300 <= freq <= 2000:
                freq_gain[i] = 1.10
                voice_band[i] = True
            elif 2000 < freq <= 4000:
                freq_gain[i] = 1.20
                voice_band[i] = True
            elif 4000 < freq <= 6000:
                freq_gain[i] = 1.08
                voice_band[i] = True
            elif 80 <= freq < 300:
                freq_gain[i] = 0.85
            elif 6000 < freq <= 8000:
                freq_gain[i] = 0.70

        low_db = db < -45
        mid_db = (db >= -45) & (db < -30)

        non_voice_extra = np.ones(num_frames)
        non_voice_extra[low_db] = 0.70
        non_voice_extra[mid_db] = 0.85

        gain_matrix = np.ones((len(freq_bins), num_frames))
        gain_matrix[voice_band, :] = freq_gain[voice_band, np.newaxis]
        gain_matrix[~voice_band, :] = freq_gain[~voice_band, np.newaxis] * non_voice_extra[np.newaxis, :]

        enhanced_magnitude = magnitude * gain_matrix
        enhanced_stft = enhanced_magnitude * phase
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=hop_length)

        if len(enhanced_audio) > len(audio_data):
            enhanced_audio = enhanced_audio[:len(audio_data)]
        elif len(enhanced_audio) < len(audio_data):
            padding = np.zeros(len(audio_data) - len(enhanced_audio))
            enhanced_audio = np.concatenate([enhanced_audio, padding])

        return enhanced_audio
    except Exception as e:
        print(f"清晰度增强失败: {e}")
        return audio_data


def apply_bluetooth_optimization(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    try:
        audio_normalized = (
            audio_data / np.max(np.abs(audio_data))
            if np.max(np.abs(audio_data)) > 0
            else audio_data
        )

        target_sample_rate = 44100
        if sample_rate > target_sample_rate:
            audio_normalized = librosa.resample(
                audio_normalized, orig_sr=sample_rate, target_sr=target_sample_rate
            )
            sample_rate = target_sample_rate

        return audio_normalized, sample_rate
    except Exception as e:
        print(f"蓝牙优化失败: {e}")
        return audio_data, sample_rate


def apply_vad_gate(
    audio_data: np.ndarray,
    sample_rate: int,
    voice_gain_db: float = 3.0,
    noise_attenuation_db: float = -3.0,
) -> tuple:
    try:
        from .vad import detect_voice_segments

        voice_segments = detect_voice_segments(
            audio_data,
            sample_rate,
            min_speech_duration=0.2,
            min_silence_duration=0.3,
        )

        if not voice_segments:
            return audio_data, []

        voice_gain = 10 ** (voice_gain_db / 20)
        noise_gain = 10 ** (noise_attenuation_db / 20)

        gated_audio = audio_data.copy() * noise_gain

        for start, end in voice_segments:
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)

            segment_length = end_sample - start_sample
            fade_samples = min(int(sample_rate * 0.05), segment_length // 4)

            voice_segment = audio_data[start_sample:end_sample]

            if segment_length > fade_samples * 2:
                fade_in = np.linspace(noise_gain, voice_gain, fade_samples)
                fade_out = np.linspace(voice_gain, noise_gain, fade_samples)

                voice_segment[:fade_samples] = (
                    audio_data[start_sample : start_sample + fade_samples] * fade_in
                )
                voice_segment[-fade_samples:] = (
                    audio_data[end_sample - fade_samples : end_sample] * fade_out
                )
                voice_segment[fade_samples:-fade_samples] = (
                    audio_data[start_sample + fade_samples : end_sample - fade_samples] * voice_gain
                )
            else:
                voice_segment = audio_data[start_sample:end_sample] * voice_gain

            gated_audio[start_sample:end_sample] = voice_segment

        max_val = np.max(np.abs(gated_audio))
        if max_val > 0.95:
            gated_audio = gated_audio * 0.95 / max_val

        return gated_audio, voice_segments
    except Exception as e:
        print(f"VAD门控失败: {e}")
        return audio_data, []


def process_single_cycling_chunk(
    audio_chunk,
    sample_rate,
    noise_reduction,
    silence_threshold,
    min_silence_duration,
    highpass_cutoff,
):
    silence_count = 0
    try:
        if noise_reduction > 0:
            try:
                import noisereduce as nr

                reduced_chunk = nr.reduce_noise(
                    y=audio_chunk,
                    sr=sample_rate,
                    prop_decrease=noise_reduction,
                    stationary=False,
                )
                audio_chunk = audio_chunk * (1 - noise_reduction) + reduced_chunk * noise_reduction
                print(f"[Cycling] 降噪完成 (stationary=False, prop_decrease={noise_reduction})")
            except Exception as e:
                print(f"[Cycling] 降噪失败: {e}")

        audio_chunk = apply_preemphasis(audio_chunk)
        audio_chunk = apply_bandpass_filter(audio_chunk, sample_rate)
        audio_chunk = apply_voice_enhancement(audio_chunk, sample_rate)
        audio_chunk = apply_dynamic_range_compression(audio_chunk, sample_rate)
        audio_chunk = apply_intelligibility_boost(audio_chunk, sample_rate)

        audio_chunk = apply_vad_gate(audio_chunk, sample_rate, voice_gain_db=8.0, noise_attenuation_db=-6.0)
        print("[Cycling] VAD门控完成")

        voice_segments = detect_voice_activity(
            audio_chunk, sample_rate, silence_threshold, min_silence_duration
        )

        if len(voice_segments) > 0:
            result_segments = []

            for start, end in voice_segments:
                start_sample = int(start * sample_rate)
                end_sample = int(end * sample_rate)

                segment = audio_chunk[start_sample:end_sample]

                segment = apply_fade(segment, sample_rate, soft_boundary_ms=50)

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
    progress_callback=None,
) -> Tuple[np.ndarray, Dict[str, Any], int]:
    def update_progress(pct, msg, status=None):
        if progress_callback:
            progress_callback(pct, msg, status)

    stats = {
        'original_duration': round(len(audio_data) / sample_rate, 2),
        'processed_duration': 0,
        'silence_segments_removed': 0,
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

            processed_chunk, silence_count = process_single_cycling_chunk(
                chunk,
                sample_rate,
                noise_reduction,
                silence_threshold,
                min_silence_duration,
                highpass_cutoff,
            )
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

        success = apply_loudnorm(temp_path, normalized_path, target_db=target_db, timeout=300)

        if success:
            audio_data, _ = librosa.load(normalized_path, sr=sample_rate, mono=True)

        if os.path.exists(temp_path):
            os.unlink(temp_path)
        if os.path.exists(normalized_path):
            os.unlink(normalized_path)

    stats['processed_duration'] = round(len(audio_data) / sample_rate, 2)

    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_path = temp_wav.name
    temp_wav.close()

    import soundfile as sf

    sf.write(temp_path, audio_data, sample_rate)

    return audio_data, stats, temp_path
