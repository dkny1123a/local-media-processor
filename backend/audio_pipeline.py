"""
音频处理管道 - 可复用的完整音频处理流程

把同步路径（main.py 的 process_audio_background）和异步路径
（task_queue.py 的 _execute_audio_processing）共用的核心逻辑提取到这里，
确保两条路径都走完整的处理流程（scene、自适应参数、自学习、FRCRN、VAD、
蓝牙优化、loudnorm 等）。
"""


def run_audio_pipeline(
    input_path: str,
    output_path: str,
    scene: str = 'default',
    auto_detect: bool = True,
    noise_reduction: float = 0.0,
    silence_threshold: float = -40.0,
    min_silence_duration: float = 0.5,
    max_volume: bool = True,
    stationary_noise: bool = False,
    task_id: str = None,
    progress_callback=None,
    task_name: str = None,
) -> dict:
    """
    完整音频处理管道（可复用）

    progress_callback 签名: callback(status: str, message: str, percentage: int)
    返回结果字典：{
        'success': bool,
        'message': str,
        'file_type': 'audio',
        'processed_audio_file': str,
        'output_path': str,
        'processed_info': dict,
        'stats': dict,
        'analysis': dict,
        'applied_params': dict,
    }
    """
    import os
    import librosa
    import numpy as np
    import tempfile
    import subprocess
    import gc

    from .adaptive_processor import analyze_audio_characteristics, calculate_adaptive_parameters
    from .audio_chunk_processor import process_audio_chunks, get_audio_duration
    from .core import convert_to_wav, load_audio_chunk
    from .video_processor import get_audio_info

    # 日志标签与前缀，保持与原 main.py 的 "[Task {task_id}]" 输出风格一致
    log_tag = task_name or (f"Task {task_id}" if task_id else "audio_pipeline")
    log_prefix = f"[{log_tag}]"

    # 进度回调缺省为空操作，避免调用方未传入时出错
    if progress_callback is None:
        def progress_callback(status, message, percentage):
            pass

    print(f"{log_prefix} 开始后台处理: {os.path.basename(input_path)}")

    try:
        print(f"{log_prefix} 转换格式: {input_path}")
        progress_callback('converting', '正在转换音频格式...', 10)
        converted_path, was_converted = convert_to_wav(input_path)

        print(f"{log_prefix} 获取音频信息")
        progress_callback('loading', '正在获取音频信息...', 15)

        try:
            total_duration = librosa.get_duration(path=converted_path)
            sr_result = librosa.get_samplerate(converted_path)
        except:
            total_duration = 300
            sr_result = 44100

        sample_rate = sr_result
        original_duration = total_duration
        print(f"{log_prefix} 音频时长: {original_duration:.2f}秒, 采样率: {sample_rate}")

        chunk_duration = 60
        chunk_size = int(sample_rate * chunk_duration)
        num_chunks = int(np.ceil(total_duration / chunk_duration))

        progress_callback('analyzing', '正在分析音频特征（噪声等级、动态范围）...', 20)

        chunk_generator, _, total_samples, _ = _load_audio_chunks(converted_path, sample_rate, chunk_duration)

        analysis = None
        for i, chunk in enumerate(chunk_generator()):
            if i == 0:
                analysis = analyze_audio_characteristics(chunk, sample_rate)
                print(f"{log_prefix} 分析完成: noise_floor={analysis['noise_floor_db']:.1f}dB, snr={analysis['signal_to_noise_ratio']:.1f}")
            break

        if was_converted and os.path.exists(converted_path):
            os.unlink(converted_path)

        adaptive_params = None
        if auto_detect and analysis:
            adaptive_params = calculate_adaptive_parameters(analysis, scene)
            noise_reduction = adaptive_params['noise_reduction']
            silence_threshold = adaptive_params['silence_threshold']
            min_silence_duration = adaptive_params['min_silence_duration']
            target_db = adaptive_params['target_db']
            highpass_cutoff = adaptive_params['highpass_cutoff']
            print(f"{log_prefix} 自适应参数: nr={noise_reduction}, hp={highpass_cutoff}, st={silence_threshold}")
        else:
            highpass_cutoff = 100.0 if scene == 'cycling' else 0.0
            target_db = -3.0 if scene in ['bluetooth', 'cycling'] else -1.0

        try:
            from .self_learning import learn_optimal_parameters, record_processing

            audio_features = {}
            if analysis:
                audio_features = {
                    'noise_floor_db': analysis['noise_floor_db'],
                    'signal_peak_db': analysis.get('signal_peak_db', -30.0),
                    'dynamic_range': analysis.get('dynamic_range', 20.0),
                    'spectral_flatness': analysis.get('spectral_flatness', 0.5),
                    'spectral_centroid': analysis.get('spectral_centroid', 1000.0),
                    'sample_rate': sample_rate,
                }

            learned_params = learn_optimal_parameters(audio_features, scene)
            if learned_params:
                if 'noise_reduction' in learned_params:
                    noise_reduction = learned_params['noise_reduction']
                if 'highpass_cutoff' in learned_params:
                    highpass_cutoff = learned_params['highpass_cutoff']
                # silence_threshold 经自学习合理性校验后才允许覆盖（见 self_learning.py）
                if 'silence_threshold' in learned_params:
                    silence_threshold = learned_params['silence_threshold']
                print(f"{log_prefix} 自学习参数覆盖: nr={noise_reduction}, hp={highpass_cutoff}, st={silence_threshold}")

            # min_silence_duration 固定 1.5s（基准线匹配），
            # 不再被自适应或自学习覆盖
            min_silence_duration = 1.5
        except Exception as e:
            print(f"{log_prefix} 自学习跳过: {e}")

        converted_path_again, was_converted_again = convert_to_wav(input_path)

        temp_wav_path = None
        cycling_stats = None

        # 修复原代码 line 282 的误导信息：根据 scene 显示不同提示
        if scene in ('cycling', 'cycling_bluetooth', 'bluetooth'):
            processing_msg = '骑行+蓝牙场景专用处理...'
        else:
            processing_msg = '音频分块处理...'
        progress_callback('processing', processing_msg, 25)

        def _chunk_progress_callback(pct, msg, status=None):
            chunk_progress = 25 + int(pct * 0.55)
            if chunk_progress > 80:
                chunk_progress = 80
            progress_callback('processing', msg, chunk_progress)

        # 修复原代码 line 296 的 scene 硬编码 bug：根据传入 scene 决定
        if scene in ('cycling', 'cycling_bluetooth', 'bluetooth'):
            chunks_scene = 'cycling_bluetooth'
        else:
            chunks_scene = scene

        try:
            processed_audio, cycling_stats, temp_wav_path = process_audio_chunks(
                converted_path_again, sample_rate, chunk_duration,
                highpass_cutoff, noise_reduction, silence_threshold, min_silence_duration,
                progress_callback=_chunk_progress_callback,
                task_name=log_tag,
                scene=chunks_scene,
                adaptive_chunk=False
            )
        except Exception as e:
            print(f"{log_prefix} 分块处理失败: {e}")
            raise

        progress_callback('processing', '蓝牙优化（降采样至16kHz）...', 85)

        optimized_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        optimized_wav.close()
        optimized_path = optimized_wav.name

        target_sample_rate = 16000
        if sample_rate > target_sample_rate:
            command = [
                'ffmpeg',
                '-i', temp_wav_path,
                '-ac', '1',
                '-ar', str(target_sample_rate),
                '-y',
                '-loglevel', 'quiet',
                optimized_path
            ]

            try:
                subprocess.run(command, check=True, capture_output=True, timeout=300)
                os.unlink(temp_wav_path)
                temp_wav_path = optimized_path
                sample_rate = target_sample_rate
                print(f"{log_prefix} 降采样完成: {sample_rate}Hz")
            except subprocess.TimeoutExpired:
                print(f"{log_prefix} 降采样超时")
                os.unlink(optimized_path)
            except Exception as e:
                print(f"{log_prefix} 降采样失败: {e}")
                os.unlink(optimized_path)
        else:
            print(f"{log_prefix} 采样率{sample_rate}Hz无需降采样")

        if was_converted_again and os.path.exists(converted_path_again):
            os.unlink(converted_path_again)

        if max_volume and temp_wav_path:
            progress_callback('processing', '正在调整音量...', 75)
            normalized_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            normalized_wav.close()
            normalized_path = normalized_wav.name

            command = [
                'ffmpeg',
                '-i', temp_wav_path,
                '-af', f'loudnorm=I={target_db}:LRA=11:TP=-1.5',
                '-y',
                '-loglevel', 'quiet',
                normalized_path
            ]

            try:
                subprocess.run(command, check=True, capture_output=True, timeout=300)
                os.unlink(temp_wav_path)
                temp_wav_path = normalized_path
            except subprocess.TimeoutExpired:
                print(f"{log_prefix} 音量调整超时")
                os.unlink(normalized_path)
            except:
                os.unlink(normalized_path)

        silence_segments_removed = cycling_stats.get('silence_segments_removed', 0) if cycling_stats else 0
        non_voice_segments_removed = cycling_stats.get('non_voice_segments_removed', 0) if cycling_stats else 0

        adaptive_result = {
            'success': True,
            'analysis': analysis,
            'applied_params': {
                'noise_reduction': noise_reduction,
                'silence_threshold': silence_threshold,
                'min_silence_duration': min_silence_duration,
                'target_db': target_db,
                'stationary_noise': stationary_noise,
                'highpass_cutoff': highpass_cutoff,
                'auto_detect': auto_detect,
                'scene': scene
            },
            'stats': {
                'duration': round(len(processed_audio) / sample_rate, 2) if len(processed_audio) > 0 else 0,
                'sample_rate': sample_rate,
                'silence_segments_removed': silence_segments_removed,
                'non_voice_segments_removed': non_voice_segments_removed
            }
        }

        if not adaptive_result['success']:
            print(f"{log_prefix} 自适应处理失败: {adaptive_result['message']}")
            return {
                "success": False,
                "message": adaptive_result['message'],
                "file_type": "audio"
            }

        # 统一输出 MP3 16000Hz 单声道（与基准线格式一致）
        progress_callback('encoding', '正在编码为MP3格式...', 90)

        # 强制输出路径为 .mp3 后缀
        output_base = os.path.splitext(output_path)[0]
        if not output_base.endswith('_processed'):
            output_base = output_base + '_processed'
        mp3_output_path = output_base + '.mp3'
        if output_path != mp3_output_path:
            output_path = mp3_output_path

        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        # 统一编码参数：MP3, 16000Hz, 单声道
        encode_sample_rate = 16000

        if temp_wav_path and os.path.exists(temp_wav_path):
            command = [
                'ffmpeg',
                '-i', temp_wav_path,
                '-ac', '1',
                '-ar', str(encode_sample_rate),
                '-c:a', 'libmp3lame',
                '-q:a', '2',
                '-y',
                output_path
            ]

            subprocess.run(command, check=True, capture_output=True, timeout=600)

            os.unlink(temp_wav_path)
        else:
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_path = temp_wav.name
            temp_wav.close()

            import soundfile as sf
            sf.write(temp_path, processed_audio, sample_rate)

            command = [
                'ffmpeg',
                '-i', temp_path,
                '-ac', '1',
                '-ar', str(encode_sample_rate),
                '-c:a', 'libmp3lame',
                '-q:a', '2',
                '-y',
                output_path
            ]

            subprocess.run(command, check=True, capture_output=True, timeout=600)

            os.unlink(temp_path)

        processed_info = get_audio_info(output_path)
        processed_duration = processed_info.get('duration', len(processed_audio) / sample_rate)
        duration_reduction = ((original_duration - processed_duration) / original_duration * 100) if original_duration > 0 else 0
        processed_info["stats"] = adaptive_result.get("stats", {})

        result = {
            "success": True,
            "message": "音频处理完成",
            "file_type": "audio",
            "processed_audio_file": os.path.basename(output_path),
            "output_path": output_path,
            "processed_info": processed_info,
            "stats": {
                "original_duration": round(original_duration, 2),
                "processed_duration": round(processed_duration, 2),
                "silence_segments_removed": adaptive_result['stats'].get('silence_segments_removed', 0),
                "duration_reduction_percent": round(duration_reduction, 2),
                "noise_reduction": adaptive_result['applied_params'].get('noise_reduction', noise_reduction),
                "sample_rate": sample_rate
            },
            "analysis": adaptive_result.get("analysis", {}),
            "applied_params": adaptive_result.get("applied_params", {})
        }

        print(f"{log_prefix} 处理完成: {output_path}")

        try:
            from .self_learning import record_processing
            processing_features = {
                'noise_floor_db': analysis['noise_floor_db'] if analysis else -50.0,
                'signal_peak_db': analysis.get('signal_peak_db', -30.0) if analysis else -30.0,
                'dynamic_range': analysis.get('dynamic_range', 20.0) if analysis else 20.0,
                'spectral_flatness': audio_features.get('spectral_flatness', 0.5) if 'audio_features' in dir() else 0.5,
                'spectral_centroid': audio_features.get('spectral_centroid', 1000.0) if 'audio_features' in dir() else 1000.0,
                'noise_level': analysis.get('noise_level', 'medium') if analysis else 'medium',
            }
            processing_params = {
                'scene': scene,
                'noise_reduction': noise_reduction,
                'silence_threshold': silence_threshold,
                'min_silence_duration': min_silence_duration,
                'highpass_cutoff': highpass_cutoff,
                'target_db': target_db,
            }
            processing_result = {
                'original_duration': original_duration,
                'processed_duration': processed_duration,
                'kept_ratio': processed_duration / original_duration if original_duration > 0 else 0,
                'silence_removed_ratio': (original_duration - processed_duration) / original_duration if original_duration > 0 else 0,
                'duration_reduction_percent': duration_reduction,
            }
            record_processing(input_path, processing_features, processing_params, processing_result)
        except Exception as e:
            print(f"{log_prefix} 自学习记录失败: {e}")

        return result

    except Exception as e:
        print(f"{log_prefix} 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"音频处理失败: {str(e)}",
            "file_type": "audio"
        }


def _load_audio_chunks(file_path, sample_rate, chunk_duration=60):
    """加载音频分块（与 main.py 中的 load_audio_chunks 保持一致）"""
    from .audio_chunk_processor import get_audio_duration
    from .core import load_audio_chunk

    total_duration = get_audio_duration(file_path)
    if total_duration <= 0:
        import librosa
        total_duration = librosa.get_duration(path=file_path)

    total_samples = int(total_duration * sample_rate)
    chunk_size = int(sample_rate * chunk_duration)
    num_chunks = (total_samples + chunk_size - 1) // chunk_size

    def chunk_generator():
        for i in range(num_chunks):
            offset = i * chunk_duration
            chunk = load_audio_chunk(file_path, sample_rate, offset, chunk_duration)
            if chunk is not None:
                yield chunk
                del chunk

    return chunk_generator, num_chunks, total_samples, sample_rate
