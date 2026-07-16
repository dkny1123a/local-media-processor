import os
import subprocess
import numpy as np
import librosa
import tempfile
import multiprocessing as mp
import threading
import time
import gc

from .core import (
    detect_silence,
    get_audio_duration,
    load_audio_chunk,
    remove_silence,
)


def detect_non_voice_segments(audio_data, sample_rate, min_duration=0.5, flatness_threshold=0.25):
    try:
        n_fft = int(sample_rate * 0.02)
        hop_length = int(sample_rate * 0.01)
        
        spectral_flatness = librosa.feature.spectral_flatness(y=audio_data, 
                                                              n_fft=n_fft, 
                                                              hop_length=hop_length)[0]
        
        non_voice_mask = spectral_flatness > flatness_threshold
        non_voice_ratio = float(np.mean(non_voice_mask))
        
        non_voice_segments = []
        in_non_voice = False
        start_frame = 0
        
        for j, is_non_voice in enumerate(non_voice_mask):
            if is_non_voice and not in_non_voice:
                in_non_voice = True
                start_frame = j
            elif not is_non_voice and in_non_voice:
                in_non_voice = False
                end_frame = j
                duration = (end_frame - start_frame) * hop_length / sample_rate
                if duration >= min_duration:
                    non_voice_segments.append((start_frame * hop_length / sample_rate, 
                                              end_frame * hop_length / sample_rate))
        
        if in_non_voice:
            end_frame = len(non_voice_mask)
            duration = (end_frame - start_frame) * hop_length / sample_rate
            if duration >= min_duration:
                non_voice_segments.append((start_frame * hop_length / sample_rate, 
                                          len(audio_data) / sample_rate))
        
        non_voice_count = len(non_voice_segments)
        if non_voice_count > 0 or non_voice_ratio > 0.05:
            print(f"[Non-Voice] 平坦度范围: [{np.min(spectral_flatness):.3f}, {np.max(spectral_flatness):.3f}], "
                  f"阈值={flatness_threshold}, 无人声比例={non_voice_ratio*100:.1f}%, 检测到{non_voice_count}个无人声段")
        
        return non_voice_segments
    except Exception as e:
        print(f"无人声检测失败: {e}")
        return []


def process_single_chunk(audio_chunk, highpass_cutoff, noise_reduction, 
                        silence_threshold, min_silence_duration, sample_rate,
                        scene='default', adaptive_chunk=False):
    silence_count = 0
    non_voice_count = 0
    from .adaptive_processor import apply_highpass_filter
    
    try:
        chunk_silence_threshold = silence_threshold
        chunk_noise_reduction = noise_reduction
        chunk_min_silence_duration = min_silence_duration
        chunk_highpass_cutoff = highpass_cutoff
        
        if adaptive_chunk and scene not in ['cycling', 'cycling_bluetooth']:
            from .adaptive_processor import analyze_audio_characteristics, calculate_adaptive_parameters
            
            chunk_analysis = analyze_audio_characteristics(audio_chunk, sample_rate)
            chunk_params = calculate_adaptive_parameters(chunk_analysis)
            
            chunk_silence_threshold = chunk_params['silence_threshold']
            chunk_noise_reduction = chunk_params['noise_reduction']
            chunk_min_silence_duration = chunk_params['min_silence_duration']
            chunk_highpass_cutoff = chunk_params['highpass_cutoff']
        
        if scene == 'cycling' or scene == 'cycling_bluetooth':
            try:
                from .ai_noise_reducer import adaptive_denoise
                audio_chunk = adaptive_denoise(audio_chunk, sample_rate)
            except Exception as e:
                print(f"[FRCRN] 自适应降噪失败: {e}")
        
        if chunk_noise_reduction > 0:
            try:
                from .core.audio_analyzer import analyze_audio_spectrum
                spectrum = analyze_audio_spectrum(audio_chunk, sample_rate)
                if spectrum.get('noise_level') != 'low':
                    import noisereduce as nr
                    reduced_chunk = nr.reduce_noise(
                        y=audio_chunk,
                        sr=sample_rate,
                        prop_decrease=chunk_noise_reduction,
                        stationary=False
                    )
                    audio_chunk = audio_chunk * (1 - chunk_noise_reduction) + reduced_chunk * chunk_noise_reduction
                    del reduced_chunk
                    gc.collect()
                else:
                    print(f"[Denoise] 低噪声环境，跳过传统降噪")
            except Exception:
                pass
        
        if scene == 'cycling' or scene == 'cycling_bluetooth':
            from .cycling_audio_processor import apply_bandpass_filter, apply_voice_enhancement
            from .cycling_audio_processor import apply_dynamic_range_compression, apply_intelligibility_boost
            from .cycling_audio_processor import apply_vad_gate, apply_preemphasis
            
            audio_chunk = apply_preemphasis(audio_chunk)
            audio_chunk = apply_bandpass_filter(audio_chunk, sample_rate)
            audio_chunk = apply_voice_enhancement(audio_chunk, sample_rate)
            audio_chunk = apply_dynamic_range_compression(audio_chunk, sample_rate)
            audio_chunk = apply_intelligibility_boost(audio_chunk, sample_rate)
            
            pre_vad_duration = len(audio_chunk) / sample_rate
            audio_chunk, voice_segments = apply_vad_gate(audio_chunk, sample_rate, voice_gain_db=8.0, noise_attenuation_db=-6.0)
            
            total_duration = len(audio_chunk) / sample_rate
            non_voice_segments = []
            last_end = 0.0
            vad_removed_duration = 0.0
            
            if voice_segments:
                voice_total_duration = sum(end - start for start, end in voice_segments)
                voice_ratio = voice_total_duration / pre_vad_duration if pre_vad_duration > 0 else 0
                
                print(f"[VAD] 语音占比: {voice_ratio*100:.1f}%, 检测到{len(voice_segments)}个语音段, 语音时长={voice_total_duration:.1f}s")
                
                if voice_ratio < 0.05:
                    print(f"[VAD] 警告: 语音占比仅{voice_ratio*100:.1f}%, 跳过VAD移除")
                    non_voice_segments = []
                else:
                    for start, end in sorted(voice_segments):
                        if start > last_end + 0.01:
                            non_voice_duration = start - last_end
                            if non_voice_duration >= max(chunk_min_silence_duration, 1.0):
                                non_voice_segments.append((last_end, start))
                                vad_removed_duration += non_voice_duration
                        last_end = end
                    
                    if last_end < total_duration - 0.01:
                        non_voice_duration = total_duration - last_end
                        if non_voice_duration >= max(chunk_min_silence_duration, 1.0):
                            non_voice_segments.append((last_end, total_duration))
                            vad_removed_duration += non_voice_duration
            else:
                print(f"[VAD] 未检测到语音段")
            
            non_voice_count = len(non_voice_segments)
            
            if non_voice_count > 0:
                print(f"[VAD] 移除{non_voice_count}个无人声段, 总时长={vad_removed_duration:.1f}s")
                
                soft_boundary_ms = 50
                soft_boundary_samples = int(sample_rate * soft_boundary_ms / 1000)
                fade_in = np.linspace(0, 1, soft_boundary_samples)
                fade_out = np.linspace(1, 0, soft_boundary_samples)
                
                voice_segments_to_keep = []
                last_end = 0
                
                for start, end in sorted(non_voice_segments):
                    if start > last_end:
                        voice_segments_to_keep.append((last_end, start))
                    last_end = end
                
                if last_end < len(audio_chunk) / sample_rate:
                    voice_segments_to_keep.append((last_end, len(audio_chunk) / sample_rate))
                
                if voice_segments_to_keep:
                    result_segments = []
                    for start, end in voice_segments_to_keep:
                        start_sample = int(start * sample_rate)
                        end_sample = int(end * sample_rate)
                        
                        segment = audio_chunk[start_sample:end_sample]
                        
                        if len(segment) > soft_boundary_samples * 2:
                            segment[:soft_boundary_samples] = segment[:soft_boundary_samples] * fade_in
                            segment[-soft_boundary_samples:] = segment[-soft_boundary_samples:] * fade_out
                        
                        result_segments.append(segment)
                    
                    audio_chunk = np.concatenate(result_segments)
            
            post_vad_duration = len(audio_chunk) / sample_rate
            print(f"[VAD] 处理前={pre_vad_duration:.1f}s, 处理后={post_vad_duration:.1f}s, VAD移除={vad_removed_duration:.1f}s")
        elif chunk_highpass_cutoff > 0:
            audio_chunk = apply_highpass_filter(audio_chunk, sample_rate, chunk_highpass_cutoff)

        pre_silence_duration = len(audio_chunk) / sample_rate
        
        if scene not in ['cycling', 'cycling_bluetooth']:
            silence_segments = detect_silence(
                audio_chunk,
                sample_rate,
                threshold_dbfs=chunk_silence_threshold,
                min_silence_duration=chunk_min_silence_duration,
            )
        else:
            silence_segments = detect_silence(
                audio_chunk,
                sample_rate,
                threshold_dbfs=-35.0,
                min_silence_duration=5.0,
            )

        silence_count = len(silence_segments)
        silence_removed_duration = sum(end - start for start, end in silence_segments)

        db = None
        if silence_count > 0:
            import librosa
            frame_length = int(sample_rate * 0.02)
            hop_length = int(sample_rate * 0.01)
            rms = librosa.feature.rms(y=audio_chunk, frame_length=frame_length, hop_length=hop_length)
            db = librosa.amplitude_to_db(rms, ref=1.0)
            threshold = chunk_silence_threshold if scene not in ['cycling', 'cycling_bluetooth'] else -40.0
            silence_ratio = float(np.mean(db < threshold))
            if silence_ratio > 0.1:
                print(f"[Silence] dB范围: [{np.min(db):.1f}, {np.max(db):.1f}], 阈值={threshold}, 静音比例={silence_ratio*100:.1f}%, 检测到{silence_count}个静音段, 移除时长={silence_removed_duration:.1f}s")

        if silence_segments:
            audio_chunk = remove_silence(audio_chunk, sample_rate, silence_segments)
        
        post_silence_duration = len(audio_chunk) / sample_rate
        if silence_removed_duration > 0:
            print(f"[Silence] 处理前={pre_silence_duration:.1f}s, 处理后={post_silence_duration:.1f}s, 移除={silence_removed_duration:.1f}s")

        return audio_chunk, silence_count, non_voice_count
    except Exception:
        return audio_chunk, 0, 0


def _chunk_worker(args):
    i, audio_chunk, highpass_cutoff, noise_reduction, silence_threshold, \
        min_silence_duration, sample_rate, result_path, scene, adaptive_chunk = args
    
    result, silence_count, non_voice_count = process_single_chunk(audio_chunk, highpass_cutoff, noise_reduction,
                                     silence_threshold, min_silence_duration, sample_rate, scene, adaptive_chunk)
    np.save(result_path, result)
    with open(result_path + '_silence', 'w') as f:
        f.write(f"{silence_count},{non_voice_count}")


def process_chunk_with_timeout(i, audio_chunk, highpass_cutoff, noise_reduction, 
                               silence_threshold, min_silence_duration, sample_rate, result_path, 
                               scene='default', adaptive_chunk=False):
    args = (i, audio_chunk, highpass_cutoff, noise_reduction, silence_threshold, 
            min_silence_duration, sample_rate, result_path, scene, adaptive_chunk)
    
    p = mp.Process(target=_chunk_worker, args=(args,))
    p.start()
    p.join(timeout=120)
    
    if p.is_alive():
        p.terminate()
        p.join()
        return False, audio_chunk, 0, 0
    
    silence_count = 0
    non_voice_count = 0
    silence_path = result_path + '_silence'
    if os.path.exists(silence_path):
        try:
            with open(silence_path, 'r') as f:
                content = f.read().strip()
                if ',' in content:
                    silence_count, non_voice_count = map(int, content.split(','))
                else:
                    silence_count = int(content)
            os.unlink(silence_path)
        except:
            pass
    
    if os.path.exists(result_path):
        try:
            result = np.load(result_path)
            os.unlink(result_path)
            return True, result, silence_count, non_voice_count
        except:
            os.unlink(result_path)
            return False, audio_chunk, 0, 0
    
    return False, audio_chunk, 0, 0


def write_single_chunk_to_wav(chunk, sample_rate, output_path):
    import soundfile as sf
    sf.write(output_path, chunk, sample_rate)


def merge_wav_files(wav_files, output_path):
    if len(wav_files) == 0:
        return False
    
    if len(wav_files) == 1:
        import shutil
        shutil.copy(wav_files[0], output_path)
        return True
    
    list_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    list_path = list_file.name
    try:
        for wav_file in wav_files:
            list_file.write(f"file '{wav_file}'\n")
    finally:
        list_file.close()
    
    if not os.path.exists(list_path):
        print(f"[Merge] 临时列表文件创建失败: {list_path}")
        return False
    
    command = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', list_path,
        '-c', 'copy',
        '-y',
        '-loglevel', 'quiet',
        output_path
    ]
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, timeout=300)
        os.unlink(list_path)
        return True
    except subprocess.TimeoutExpired:
        print(f"[Merge] ffmpeg合并超时")
        os.unlink(list_path)
        return False
    except subprocess.CalledProcessError as e:
        print(f"[Merge] ffmpeg合并失败: {e}")
        print(f"[Merge] stderr: {e.stderr.decode('utf-8', errors='ignore')}")
        os.unlink(list_path)
        return False
    except Exception as e:
        print(f"[Merge] 合并异常: {e}")
        if os.path.exists(list_path):
            os.unlink(list_path)
        return False


def process_audio_chunks(file_path, sample_rate, chunk_duration, highpass_cutoff, 
                         noise_reduction, silence_threshold, min_silence_duration,
                         progress_callback=None, task_name="audio", scene='default',
                         adaptive_chunk=False):
    total_duration = get_audio_duration(file_path)
    if total_duration <= 0:
        try:
            total_duration = librosa.get_duration(path=file_path)
        except:
            total_duration = 0
    
    if total_duration <= 0:
        return np.array([]), {"error": "无法获取音频时长"}
    
    num_chunks = int(np.ceil(total_duration / chunk_duration))
    temp_wav_files = []
    current_batch = []
    BATCH_SIZE = 10
    
    stats = {
        "total_chunks": num_chunks,
        "processed_chunks": 0,
        "skipped_chunks": 0,
        "timeout_chunks": 0,
        "total_time": 0,
        "chunk_times": [],
        "temp_files_created": 0,
        "silence_segments_removed": 0,
        "non_voice_segments_removed": 0
    }
    
    start_time = time.time()
    processed_chunks = []
    
    progress_lock = threading.Lock()
    current_chunk_progress = 0
    progress_thread_running = True
    
    def progress_update_thread():
        nonlocal current_chunk_progress
        while progress_thread_running:
            time.sleep(0.5)
            if progress_callback and current_chunk_progress < 100:
                with progress_lock:
                    chunk_pct = current_chunk_progress
                progress_callback(chunk_pct, f'正在处理块 {int(chunk_pct * num_chunks / 100) + 1}/{num_chunks}...')
    
    try:
        if progress_callback:
            progress_thread = threading.Thread(target=progress_update_thread, daemon=True)
            progress_thread.start()
        
        for i in range(num_chunks):
            chunk_start_time = time.time()
            
            offset = i * chunk_duration
            
            print(f"[{task_name}] 加载块 {i+1}/{num_chunks} (offset={offset:.1f}s)")
            
            audio_chunk = load_audio_chunk(file_path, sample_rate, offset, chunk_duration)
            if audio_chunk is None or len(audio_chunk) == 0:
                print(f"[{task_name}] 加载块 {i+1} 失败，跳过")
                stats["skipped_chunks"] += 1
                with progress_lock:
                    current_chunk_progress = int(((i + 1) / num_chunks) * 100)
                continue
            
            with progress_lock:
                current_chunk_progress = int((i / num_chunks) * 100)
            
            result_file = tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
            result_file.close()
            result_path = result_file.name
            
            success, result, silence_count, non_voice_count = process_chunk_with_timeout(
                i, audio_chunk, highpass_cutoff, noise_reduction,
                silence_threshold, min_silence_duration, sample_rate, result_path, scene,
                adaptive_chunk
            )
            
            chunk_to_save = result if success else audio_chunk
            
            if success:
                stats["processed_chunks"] += 1
                stats["silence_segments_removed"] += silence_count
                stats["non_voice_segments_removed"] += non_voice_count
                print(f"[{task_name}] 块 {i+1} 静音段数: {silence_count}, 无人声段数: {non_voice_count}, "
                      f"累计静音: {stats['silence_segments_removed']}, 累计无人声: {stats['non_voice_segments_removed']}")
            else:
                stats["timeout_chunks"] += 1
                print(f"[{task_name}] 块 {i+1} 超时或失败，保留原始音频")
            
            processed_chunks.append(chunk_to_save)
            
            del audio_chunk, result, chunk_to_save
            gc.collect()
            
            if len(processed_chunks) >= BATCH_SIZE:
                if temp_wav_files:
                    prev_temp = temp_wav_files.pop()
                    merged_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    merged_temp.close()
                    merged_path = merged_temp.name
                    
                    prev_audio, _ = librosa.load(prev_temp, sr=sample_rate, mono=True)
                    os.unlink(prev_temp)
                    
                    current_batch_audio = np.concatenate(processed_chunks)
                    combined = np.concatenate([prev_audio, current_batch_audio])
                    
                    import soundfile as sf
                    sf.write(merged_path, combined, sample_rate)
                    
                    temp_wav_files.append(merged_path)
                    del prev_audio, current_batch_audio, combined
                else:
                    current_batch_audio = np.concatenate(processed_chunks)
                    batch_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    batch_temp.close()
                    batch_path = batch_temp.name
                    
                    import soundfile as sf
                    sf.write(batch_path, current_batch_audio, sample_rate)
                    
                    temp_wav_files.append(batch_path)
                    del current_batch_audio
                
                processed_chunks = []
                gc.collect()
            
            chunk_time = time.time() - chunk_start_time
            stats["chunk_times"].append(chunk_time)
            
            with progress_lock:
                current_chunk_progress = int(((i + 1) / num_chunks) * 100)
            
            print(f"[{task_name}] 块 {i+1}/{num_chunks} 完成，耗时 {chunk_time:.2f}s")
        
        if processed_chunks:
            if progress_callback:
                progress_callback(95, '正在合并处理结果...')
            if temp_wav_files:
                prev_temp = temp_wav_files.pop()
                merged_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                merged_temp.close()
                merged_path = merged_temp.name
                
                prev_audio, _ = librosa.load(prev_temp, sr=sample_rate, mono=True)
                os.unlink(prev_temp)
                
                current_batch_audio = np.concatenate(processed_chunks)
                combined = np.concatenate([prev_audio, current_batch_audio])
                
                import soundfile as sf
                sf.write(merged_path, combined, sample_rate)
                
                temp_wav_files.append(merged_path)
                del prev_audio, current_batch_audio, combined
            else:
                current_batch_audio = np.concatenate(processed_chunks)
                batch_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                batch_temp.close()
                batch_path = batch_temp.name
                
                import soundfile as sf
                sf.write(batch_path, current_batch_audio, sample_rate)
                
                temp_wav_files.append(batch_path)
                del current_batch_audio
            
            processed_chunks = []
            gc.collect()
        
        stats["total_time"] = time.time() - start_time
        
        processed_audio = np.array([])
        temp_wav_path = None
        
        if temp_wav_files:
            final_temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            final_temp_wav.close()
            temp_wav_path = final_temp_wav.name
            
            print(f"[{task_name}] 合并 {len(temp_wav_files)} 个批次文件...")
            if merge_wav_files(temp_wav_files, temp_wav_path):
                pass
            
            if progress_callback:
                progress_callback(98, '合并完成，准备输出...')
            
            for wav_file in temp_wav_files:
                os.unlink(wav_file)
            temp_wav_files = []
    
    except Exception as e:
        print(f"[{task_name}] 分块处理异常: {str(e)}")
        import traceback
        traceback.print_exc()
        
        for wav_file in current_batch:
            if os.path.exists(wav_file):
                os.unlink(wav_file)
        current_batch = []
        
        for wav_file in temp_wav_files:
            if os.path.exists(wav_file):
                os.unlink(wav_file)
        temp_wav_files = []
        
        progress_thread_running = False
        
        return np.array([]), {"error": str(e)}, None
    
    finally:
        progress_thread_running = False
    
    avg_time = np.mean(stats["chunk_times"]) if stats["chunk_times"] else 0
    max_time = np.max(stats["chunk_times"]) if stats["chunk_times"] else 0
    
    print(f"[{task_name}] 处理完成: 总块数={stats['total_chunks']}, 处理={stats['processed_chunks']}, "
          f"超时={stats['timeout_chunks']}, 跳过={stats['skipped_chunks']}, "
          f"临时文件={stats['temp_files_created']}, 总耗时={stats['total_time']:.2f}s, "
          f"平均每块={avg_time:.2f}s, 最长={max_time:.2f}s")
    
    print(f"[{task_name}] 静音段统计: 累计移除={stats['silence_segments_removed']}")
    print(f"[{task_name}] 无人声段统计: 累计移除={stats['non_voice_segments_removed']}")
    
    return processed_audio, stats, temp_wav_path