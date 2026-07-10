import subprocess
import os
import numpy as np
import librosa
import soundfile as sf
import tempfile
import multiprocessing as mp
from .adaptive_processor import process_audio_adaptive, apply_highpass_filter
from .audio_chunk_processor import process_audio_chunks, load_audio_chunk, get_audio_duration, process_chunk_with_timeout


def process_chunk_with_timeout(i, audio_chunk, highpass_cutoff, noise_reduction, silence_threshold, min_silence_duration, sample_rate, result_path):
    try:
        if highpass_cutoff > 0:
            audio_chunk = apply_highpass_filter(audio_chunk, sample_rate, highpass_cutoff)
        
        if noise_reduction > 0:
            try:
                import noisereduce as nr
                reduced_chunk = nr.reduce_noise(
                    y=audio_chunk,
                    sr=sample_rate,
                    prop_decrease=noise_reduction
                )
                audio_chunk = audio_chunk * (1 - noise_reduction) + reduced_chunk * noise_reduction
            except Exception:
                pass
        
        frame_length = int(sample_rate * 0.02)
        hop_length = int(sample_rate * 0.01)
        
        audio_normalized = audio_chunk / np.max(np.abs(audio_chunk)) if np.max(np.abs(audio_chunk)) > 0 else audio_chunk
        
        rms = librosa.feature.rms(y=audio_normalized, frame_length=frame_length, hop_length=hop_length)
        db = librosa.amplitude_to_db(rms, ref=np.max)
        
        silence_mask = db < silence_threshold
        
        silence_segments = []
        in_silence = False
        start_frame = 0
        
        for j, is_silent in enumerate(silence_mask[0]):
            if is_silent and not in_silence:
                in_silence = True
                start_frame = j
            elif not is_silent and in_silence:
                in_silence = False
                end_frame = j
                duration = (end_frame - start_frame) * hop_length / sample_rate
                if duration >= min_silence_duration:
                    silence_segments.append((start_frame * hop_length / sample_rate, 
                                           end_frame * hop_length / sample_rate))
        
        if in_silence:
            end_frame = len(silence_mask[0])
            duration = (end_frame - start_frame) * hop_length / sample_rate
            if duration >= min_silence_duration:
                silence_segments.append((start_frame * hop_length / sample_rate, 
                                       len(audio_chunk) / sample_rate))
        
        if silence_segments:
            segments_to_keep = []
            last_end = 0
            
            soft_boundary_ms = 50
            soft_boundary_samples = int(sample_rate * soft_boundary_ms / 1000)
            
            for start, end in sorted(silence_segments):
                if start > last_end:
                    segments_to_keep.append((last_end, start))
                last_end = end
            
            if last_end < len(audio_chunk) / sample_rate:
                segments_to_keep.append((last_end, len(audio_chunk) / sample_rate))
            
            if segments_to_keep:
                result_segments = []
                for start, end in segments_to_keep:
                    start_sample = int(start * sample_rate)
                    end_sample = int(end * sample_rate)
                    
                    segment = audio_chunk[start_sample:end_sample]
                    
                    if len(segment) > soft_boundary_samples * 2:
                        fade_in = np.linspace(0, 1, soft_boundary_samples)
                        fade_out = np.linspace(1, 0, soft_boundary_samples)
                        
                        segment[:soft_boundary_samples] = segment[:soft_boundary_samples] * fade_in
                        segment[-soft_boundary_samples:] = segment[-soft_boundary_samples:] * fade_out
                    
                    result_segments.append(segment)
                
                audio_chunk = np.concatenate(result_segments)
        
        np.save(result_path, audio_chunk)
    except Exception:
        np.save(result_path, audio_chunk)

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
            
            update_progress(0.25, '正在分析音频特征...', 'processing')
            
            analysis = None
            for i in range(num_chunks):
                offset = i * chunk_duration
                try:
                    from .audio_chunk_processor import load_audio_chunk
                    chunk = load_audio_chunk(audio_output_path, sample_rate, offset, chunk_duration)
                except:
                    chunk = librosa.load(audio_output_path, sr=sample_rate, mono=True, 
                                        offset=offset, duration=chunk_duration)[0]
                if i == 0:
                    from .adaptive_processor import analyze_audio_characteristics, calculate_adaptive_parameters, apply_highpass_filter
                    analysis = analyze_audio_characteristics(chunk, sample_rate)
                    print(f"[Video] 分析完成: noise_floor={analysis['noise_floor_db']:.1f}dB, snr={analysis['signal_to_noise_ratio']:.1f}")
                del chunk
                if i == 0:
                    break
            
            if auto_detect and analysis:
                adaptive_params = calculate_adaptive_parameters(analysis, scene)
                noise_reduction = adaptive_params['noise_reduction']
                silence_threshold = adaptive_params['silence_threshold']
                min_silence_duration = adaptive_params['min_silence_duration']
                target_db = adaptive_params['target_db']
                highpass_cutoff = adaptive_params['highpass_cutoff']
                print(f"[Video] 自适应参数: nr={noise_reduction}, hp={highpass_cutoff}, st={silence_threshold}")
            else:
                highpass_cutoff = 100.0 if scene == 'cycling' else 0.0
                target_db = -3.0 if scene in ['bluetooth', 'cycling'] else -1.0
            
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
                    os.unlink(normalized_path)
                except:
                    os.unlink(normalized_path)
            
            if scene in ['cycling', 'cycling_bluetooth', 'bluetooth'] and temp_wav_path:
                update_progress(0.88, '蓝牙优化（降采样至16kHz）...', 'processing')
                optimized_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                optimized_wav.close()
                optimized_path = optimized_wav.name
                
                command = [
                    'ffmpeg',
                    '-i', temp_wav_path,
                    '-ac', '1',
                    '-ar', '16000',
                    '-y',
                    '-loglevel', 'quiet',
                    optimized_path
                ]
                
                try:
                    subprocess.run(command, check=True, capture_output=True, timeout=300)
                    os.unlink(temp_wav_path)
                    temp_wav_path = optimized_path
                    sample_rate = 16000
                    print(f"[Video] 降采样完成: {sample_rate}Hz")
                except subprocess.TimeoutExpired:
                    print(f"[Video] 降采样超时")
                    os.unlink(optimized_path)
                except Exception as e:
                    print(f"[Video] 降采样失败: {e}")
                    os.unlink(optimized_path)
            
            update_progress(0.9, '正在编码为MP3格式...', 'processing')
            
            if temp_wav_path and os.path.exists(temp_wav_path):
                command = [
                    'ffmpeg',
                    '-i', temp_wav_path,
                    '-ac', '1',
                    '-ar', str(sample_rate),
                    '-c:a', 'libmp3lame',
                    '-q:a', '2',
                    '-y',
                    processed_audio_path
                ]
                
                subprocess.run(command, check=True, capture_output=True, timeout=600)
                
                os.unlink(temp_wav_path)
            else:
                temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_path = temp_wav.name
                temp_wav.close()
                
                sf.write(temp_path, processed_audio, sample_rate)
                
                command = [
                    'ffmpeg',
                    '-i', temp_path,
                    '-ac', '1',
                    '-ar', str(sample_rate),
                    '-c:a', 'libmp3lame',
                    '-q:a', '2',
                    '-y',
                    processed_audio_path
                ]
                
                subprocess.run(command, check=True, capture_output=True, timeout=600)
                
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
