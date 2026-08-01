import os
import numpy as np

_vad_model = None
_vad_initialized = False


def init_vad(model_dir=None):
    global _vad_model, _vad_initialized
    
    if _vad_initialized:
        return _vad_model is not None
    
    _vad_initialized = True
    
    try:
        from silero_vad import load_silero_vad
        try:
            import onnxruntime as ort
            _vad_model = load_silero_vad(onnx=True)
            print("[VAD] Silero VAD模型加载成功 (ONNX)")
            return True
        except ImportError:
            _vad_model = load_silero_vad()
            print("[VAD] Silero VAD模型加载成功 (JIT)")
            return True
        except Exception as e:
            _vad_model = load_silero_vad()
            print(f"[VAD] Silero VAD模型加载成功 (JIT, ONNX失败: {e})")
            return True
    except Exception as e:
        print(f"[VAD] 模型加载失败: {e}")
        _vad_model = None
        return False


def get_vad_model():
    global _vad_model, _vad_initialized
    if not _vad_initialized:
        init_vad()
    return _vad_model


def detect_voice_segments(audio_data, sample_rate, min_speech_duration=0.3, min_silence_duration=0.5,
                          threshold=0.5):
    vad_model = get_vad_model()

    if vad_model is not None:
        try:
            from silero_vad import get_speech_timestamps
            import torch

            if sample_rate not in [8000, 16000]:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000

            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # 归一化到峰值0.9，确保低音量录音也能被Silero VAD正确检测
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data * (0.9 / max_val)

            torch.set_num_threads(1)
            audio_tensor = torch.from_numpy(audio_data)

            speech_timestamps = get_speech_timestamps(
                audio_tensor,
                vad_model,
                threshold=threshold,
                min_silence_duration_ms=int(min_silence_duration * 1000),
                min_speech_duration_ms=int(min_speech_duration * 1000),
                return_seconds=True
            )

            segments = [(s['start'], s['end']) for s in speech_timestamps]

            # 回退机制：如果Silero未检测到语音但音频有能量，使用能量VAD
            if not segments:
                rms = np.sqrt(np.mean(audio_data ** 2))
                rms_db = 20 * np.log10(rms + 1e-10)
                if rms_db > -80:
                    print(f"[VAD] Silero未检测到语音(RMS={rms_db:.1f}dB)，回退到能量VAD")
                    return _energy_based_vad(audio_data, sample_rate, min_speech_duration, min_silence_duration)

            # 回退机制：如果Silero检测到语音占比过低（<10%），可能漏检低能量语音
            total_duration = len(audio_data) / sample_rate
            voice_duration = sum(end - start for start, end in segments)
            if total_duration > 0 and voice_duration / total_duration < 0.10:
                print(f"[VAD] Silero检测语音占比过低({voice_duration/total_duration*100:.1f}%)，回退到能量VAD")
                return _energy_based_vad(audio_data, sample_rate, min_speech_duration, min_silence_duration)

            return segments

        except Exception as e:
            print(f"[VAD] Silero检测失败，使用能量VAD: {e}")

    return _energy_based_vad(audio_data, sample_rate, min_speech_duration, min_silence_duration)


def _energy_based_vad(audio_data, sample_rate, min_speech_duration=0.3, min_silence_duration=0.5):
    try:
        import librosa

        frame_length = int(sample_rate * 0.03)
        hop_length = int(sample_rate * 0.01)

        rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=1.0)

        noise_db = np.percentile(rms_db, 20)
        voice_db = np.percentile(rms_db, 80)

        threshold_db = noise_db + (voice_db - noise_db) * 0.3

        if voice_db - noise_db < 3:
            print(f"[VAD-Energy] 信号动态范围太小 ({voice_db - noise_db:.1f}dB), 返回全部音频")
            return [(0.0, len(audio_data) / sample_rate)]

        is_voice = rms_db > threshold_db

        segments = []
        in_voice = False
        start_sample = 0

        for i, v in enumerate(is_voice):
            if v and not in_voice:
                in_voice = True
                start_sample = i
            elif not v and in_voice:
                in_voice = False
                start_time = start_sample * hop_length / sample_rate
                end_time = i * hop_length / sample_rate
                if end_time - start_time >= min_speech_duration:
                    segments.append((start_time, end_time))

        if in_voice:
            start_time = start_sample * hop_length / sample_rate
            end_time = len(is_voice) * hop_length / sample_rate
            if end_time - start_time >= min_speech_duration:
                segments.append((start_time, end_time))

        merged = []
        for seg in segments:
            if merged and seg[0] - merged[-1][1] < min_silence_duration:
                merged[-1] = (merged[-1][0], seg[1])
            else:
                merged.append(seg)

        print(f"[VAD-Energy] 阈值={threshold_db:.1f}dB, 噪声={noise_db:.1f}dB, 语音={voice_db:.1f}dB, "
              f"检测到{len(merged)}个语音段")

        if not merged:
            return [(0.0, len(audio_data) / sample_rate)]

        return merged

    except Exception as e:
        print(f"[VAD-Energy] 能量VAD失败: {e}")
        return [(0.0, len(audio_data) / sample_rate)]


def detect_non_voice_segments_vad(audio_data, sample_rate, min_duration=0.5):
    voice_segments = detect_voice_segments(audio_data, sample_rate, 
                                           min_speech_duration=0.3,
                                           min_silence_duration=min_duration)
    
    total_duration = len(audio_data) / sample_rate
    non_voice_segments = []
    last_end = 0.0
    
    for start, end in sorted(voice_segments):
        if start > last_end + 0.01:
            non_voice_duration = start - last_end
            if non_voice_duration >= min_duration:
                non_voice_segments.append((last_end, start))
        last_end = end
    
    if last_end < total_duration - 0.01:
        non_voice_duration = total_duration - last_end
        if non_voice_duration >= min_duration:
            non_voice_segments.append((last_end, total_duration))
    
    return non_voice_segments
