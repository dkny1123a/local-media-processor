import subprocess
import os
import numpy as np
import librosa
import soundfile as sf
import tempfile
from .adaptive_processor import apply_highpass_filter
from .audio_chunk_processor import process_audio_chunks, load_audio_chunk, get_audio_duration
from .core import apply_loudnorm, encode_to_mp3, resample_audio


def extract_audio_from_video(video_path, audio_output_path):
    try:
        command = [
            'ffmpeg',
            '-i', video_path,
            '-q:a', '0',
            '-map', 'a',
            '-y',
            '-loglevel', 'error',
            audio_output_path
        ]
        subprocess.run(command, check=True, capture_output=True, timeout=600)
        return True
    except subprocess.CalledProcessError as e:
        print(f"音频提取失败: {e.stderr.decode() if e.stderr else str(e)}")
        return False
    except subprocess.TimeoutExpired:
        print(f"音频提取超时（10分钟）: {video_path}")
        return False
    except Exception as e:
        print(f"音频提取失败: {str(e)}")
        return False

def get_audio_info(audio_path, audio_data=None, sample_rate=None):
    try:
        import subprocess as sp
        import json
        
        duration = 0
        detected_sample_rate = sample_rate if sample_rate else 44100
        
        try:
            result = sp.run(
                ['ffprobe', '-v', 'quiet', '-print_format', 'json', 
                 '-show_format', '-show_streams', audio_path],
                capture_output=True, text=True, timeout=30
            )
            probe_data = json.loads(result.stdout)
            
            for stream in probe_data.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    duration = float(stream.get('duration', probe_data.get('format', {}).get('duration', 0)))
                    detected_sample_rate = int(stream.get('sample_rate', detected_sample_rate))
                    break
            
            if duration == 0:
                duration = float(probe_data.get('format', {}).get('duration', 0))
        except Exception as e:
            print(f"ffprobe获取信息失败: {str(e)}")
        
        waveform = []
        if audio_data is not None and len(audio_data) > 0:
            samples_per_point = max(1, len(audio_data) // 200)
            for i in range(0, len(audio_data), samples_per_point):
                chunk = audio_data[i:i + samples_per_point]
                waveform.append(float(np.max(np.abs(chunk))))
        else:
            try:
                if duration > 0:
                    result = sp.run(
                        ['ffmpeg', '-i', audio_path, '-f', 's16le', '-acodec', 'pcm_s16le',
                         '-ac', '1', '-ar', '2000', '-'],
                        capture_output=True, timeout=120
                    )
                    if result.stdout and len(result.stdout) > 0:
                        audio_lowres = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
                        total_samples = len(audio_lowres)
                        num_points = 200
                        samples_per_point = max(1, total_samples // num_points)
                        for i in range(num_points):
                            start = i * samples_per_point
                            end = min(start + samples_per_point, total_samples)
                            if start < total_samples:
                                waveform.append(float(np.max(np.abs(audio_lowres[start:end]))))
                            else:
                                waveform.append(0.0)
            except Exception as e:
                print(f"生成波形图失败: {str(e)}")
                waveform = [0.0] * 200
        
        if waveform:
            max_val = max(waveform) if max(waveform) > 0 else 1
            waveform = [w / max_val for w in waveform]
        
        return {
            "duration": round(duration, 2),
            "sample_rate": detected_sample_rate,
            "waveform": waveform,
            "size_bytes": os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
        }
    except Exception as e:
        print(f"获取音频信息失败: {str(e)}")
        return {
            "duration": 0,
            "sample_rate": 0,
            "waveform": [],
            "size_bytes": 0
        }

def process_video(
    input_path, 
    audio_output_path, 
    processed_audio_path, 
    extract_audio=True, 
    do_process_audio=True,
    silence_threshold=-40.0,
    min_silence_duration=0.5,
    max_volume=True,
    noise_reduction=0.0,
    stationary_noise=False,
    auto_detect=True,
    scene="bluetooth",
    progress_callback=None
):
    def update_progress(pct, msg, status=None):
        if progress_callback:
            progress_callback(pct, msg, status)
    
    result_data = {
        "success": False,
        "message": "",
        "audio_info": None,
        "processed_info": None,
        "analysis": {},
        "applied_params": {}
    }
    
    try:
        output_dir = os.path.dirname(audio_output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        if extract_audio:
            print("[Video] 正在提取音频...")
            update_progress(0.1, '正在提取音频...', 'processing')
            success = extract_audio_from_video(input_path, audio_output_path)
            if not success:
                result_data["message"] = "音频提取失败"
                return result_data
            
            update_progress(0.2, '正在获取音频信息...', 'processing')
            result_data["audio_info"] = get_audio_info(audio_output_path)
        
        if do_process_audio and extract_audio:
            print("[Video] 获取音频信息...")
            update_progress(0.22, '正在获取音频信息...', 'processing')
            
            try:
                total_duration = get_audio_duration(audio_output_path)
                sample_rate = librosa.get_samplerate(audio_output_path)
            except:
                total_duration = 300
                sample_rate = 44100
            
            original_duration = total_duration
            print(f"[Video] 音频时长: {original_duration:.2f}秒, 采样率: {sample_rate}")
            
            chunk_duration = 30
            num_chunks = int(np.ceil(total_duration / chunk_duration))
            
            update_progress(0.25, '正在分析音频特征（噪声等级、动态范围）...', 'processing')

            # ============================================================
            # 全局分析：整条音频分段扫描，不再只取第一个 chunk
            # ============================================================
            # 原逻辑：仅取 i==0 的第一个 chunk 分析，break 退出
            #   - 前 30s 可能不代表整条音频（中途场景变化）
            #   - noise_floor 偏差导致全局阈值不准
            #
            # 新逻辑：扫描整条音频，均匀采样最多 10 段
            #   - 对所有段的 RMS dB 取第 5 百分位作为 noise_floor
            #   - 用整条音频的 noise_floor 计算全局阈值
            # ============================================================
            from .adaptive_processor import analyze_audio_characteristics, calculate_adaptive_parameters, apply_highpass_filter
            from .audio_chunk_processor import load_audio_chunk

            analysis = None
            all_rms_db_list = []
            num_analysis_segments = min(10, num_chunks)
            if num_analysis_segments < 1:
                num_analysis_segments = 1

            if num_analysis_segments == 1:
                sample_indices = [0]
            else:
                sample_indices = [int(i * (num_chunks - 1) / (num_analysis_segments - 1)) for i in range(num_analysis_segments)]

            sampled_count = 0
            for i in range(num_chunks):
                if i not in sample_indices:
                    continue
                offset = i * chunk_duration
                try:
                    chunk = load_audio_chunk(audio_output_path, sample_rate, offset, chunk_duration)
                except:
                    chunk = librosa.load(audio_output_path, sr=sample_rate, mono=True,
                                        offset=offset, duration=chunk_duration)[0]
                if sampled_count == 0:
                    analysis = analyze_audio_characteristics(chunk, sample_rate)
                frame_length = int(sample_rate * 0.02)
                hop_length = int(sample_rate * 0.01)
                rms = librosa.feature.rms(y=chunk, frame_length=frame_length, hop_length=hop_length)[0]
                rms_db = librosa.amplitude_to_db(rms, ref=1.0)
                all_rms_db_list.append(rms_db)
                sampled_count += 1
                del chunk
                if sampled_count >= num_analysis_segments:
                    break

            if all_rms_db_list:
                all_rms_db = np.concatenate(all_rms_db_list)
                global_noise_floor_db = float(np.percentile(all_rms_db, 5))
                global_signal_peak_db = float(np.percentile(all_rms_db, 95))
                global_dynamic_range = global_signal_peak_db - global_noise_floor_db

                if analysis:
                    analysis['noise_floor_db'] = global_noise_floor_db
                    analysis['signal_peak_db'] = global_signal_peak_db
                    analysis['dynamic_range'] = global_dynamic_range
                    analysis['signal_to_noise_ratio'] = global_dynamic_range

                print(f"[Video] 全局分析完成({sampled_count}段采样): "
                      f"noise_floor={global_noise_floor_db:.1f}dB, "
                      f"signal_peak={global_signal_peak_db:.1f}dB, "
                      f"dynamic_range={global_dynamic_range:.1f}dB")

            if auto_detect and analysis:
                adaptive_params = calculate_adaptive_parameters(analysis, scene)
                noise_reduction = adaptive_params['noise_reduction']
                silence_threshold = adaptive_params['silence_threshold']
                min_silence_duration = 0.8
                target_db = adaptive_params['target_db']
                highpass_cutoff = adaptive_params['highpass_cutoff']
                print(f"[Video] 自适应参数: nr={noise_reduction}, hp={highpass_cutoff}, st={silence_threshold}")
            else:
                highpass_cutoff = 100.0 if scene == 'cycling' else 0.0
                # 目标响度 -14 LUFS（骑行嘈杂环境，基于第一性原理）
                # apply_loudnorm 内部固定 volume=25dB，此值仅用于日志
                target_db = -14.0
            
            update_progress(0.3, '开始分块处理音频...', 'processing')
            
            def video_chunk_progress(pct, msg, status=None):
                update_progress(0.3 + pct * 0.005, msg, 'processing')
            
            processed_audio, stats, temp_wav_path = process_audio_chunks(
                audio_output_path, sample_rate, chunk_duration,
                highpass_cutoff, noise_reduction, silence_threshold, min_silence_duration,
                progress_callback=video_chunk_progress, task_name="Video",
                scene=scene, adaptive_chunk=True
            )
            
            if max_volume and temp_wav_path:
                update_progress(0.85, '正在调整音量...', 'processing')
                normalized_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                normalized_wav.close()
                normalized_path = normalized_wav.name

                # apply_loudnorm 内部已包含 aresample=22050，无需再单独降采样
                success = apply_loudnorm(temp_wav_path, normalized_path, target_db=target_db, timeout=300)

                if success:
                    os.unlink(temp_wav_path)
                    temp_wav_path = normalized_path
                    sample_rate = 22050  # apply_loudnorm 已降采样至此
                    print(f"[Video] 响度归一化完成（含降采样至22050Hz）")
                else:
                    if os.path.exists(normalized_path):
                        os.unlink(normalized_path)
            elif scene in ['cycling', 'cycling_bluetooth', 'bluetooth'] and temp_wav_path:
                # max_volume=False 时，apply_loudnorm 未执行，需单独降采样
                update_progress(0.88, '降采样至22.05kHz...', 'processing')
                optimized_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                optimized_wav.close()
                optimized_path = optimized_wav.name

                success = resample_audio(temp_wav_path, optimized_path, target_sample_rate=22050, timeout=300)

                if success:
                    os.unlink(temp_wav_path)
                    temp_wav_path = optimized_path
                    sample_rate = 22050
                    print(f"[Video] 降采样完成: {sample_rate}Hz")
                else:
                    print(f"[Video] 降采样失败")
                    if os.path.exists(optimized_path):
                        os.unlink(optimized_path)

            update_progress(0.9, '正在编码为MP3格式...', 'processing')

            if temp_wav_path and os.path.exists(temp_wav_path):
                success = encode_to_mp3(temp_wav_path, processed_audio_path, sample_rate, bitrate=96, timeout=600)
                if not success:
                    raise Exception("MP3编码失败")
                os.unlink(temp_wav_path)
            else:
                temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_path = temp_wav.name
                temp_wav.close()

                sf.write(temp_path, processed_audio, sample_rate)

                success = encode_to_mp3(temp_path, processed_audio_path, sample_rate, bitrate=96, timeout=600)
                if not success:
                    raise Exception("MP3编码失败")

                os.unlink(temp_path)
            
            processed_info = get_audio_info(processed_audio_path)
            result_data["processed_info"] = processed_info
            silence_segments_removed = stats.get('silence_segments_removed', 0) if stats else 0
            non_voice_segments_removed = stats.get('non_voice_segments_removed', 0) if stats else 0
            result_data["processed_info"]["stats"] = {
                'duration': processed_info.get('duration', 0),
                'sample_rate': sample_rate,
                'silence_segments_removed': silence_segments_removed,
                'non_voice_segments_removed': non_voice_segments_removed
            }
            result_data["analysis"] = analysis or {}
            
            if auto_detect and analysis:
                adaptive_params = calculate_adaptive_parameters(analysis, scene)
                stationary_noise = adaptive_params.get('stationary_noise', False)
            else:
                stationary_noise = False
            
            result_data["applied_params"] = {
                'noise_reduction': noise_reduction,
                'silence_threshold': silence_threshold,
                'min_silence_duration': min_silence_duration,
                'target_db': target_db,
                'stationary_noise': stationary_noise,
                'highpass_cutoff': highpass_cutoff,
                'auto_detect': auto_detect,
                'scene': scene
            }
        
        update_progress(1.0, '处理完成', 'complete')
        print("[Video] 处理完成")
        
        try:
            if extract_audio and audio_output_path and os.path.exists(audio_output_path):
                os.unlink(audio_output_path)
                print(f"[Video] 清理提取的音频文件: {audio_output_path}")
        except Exception as cleanup_e:
            print(f"[Video] 清理提取的音频文件失败: {cleanup_e}")
        
        result_data["success"] = True
        result_data["message"] = "视频处理完成"
        return result_data
    
    except Exception as e:
        print(f"[Video] 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        result_data["message"] = f"视频处理失败: {str(e)}"
        
        try:
            if 'temp_wav_path' in locals() and temp_wav_path and os.path.exists(temp_wav_path):
                os.unlink(temp_wav_path)
                print(f"[Video] 清理失败时的临时文件: {temp_wav_path}")
            if extract_audio and audio_output_path and os.path.exists(audio_output_path):
                os.unlink(audio_output_path)
                print(f"[Video] 清理提取的音频文件: {audio_output_path}")
        except Exception as cleanup_e:
            print(f"[Video] 清理临时文件失败: {cleanup_e}")
        
        return result_data
