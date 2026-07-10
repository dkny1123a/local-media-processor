import os
import subprocess
import numpy as np
import librosa
import tempfile
import multiprocessing as mp
import time
import gc


def get_audio_duration(file_path):
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', 
             '-of', 'csv=p=0', file_path],
            capture_output=True, text=True, check=True, timeout=30
        )
        return float(result.stdout.strip())
    except subprocess.TimeoutExpired:
        print(f"ffprobe超时: {file_path}")
    except:
        try:
            return librosa.get_duration(path=file_path)
        except:
            return 0


def load_audio_chunk_ffmpeg(file_path, sample_rate, offset, duration):
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
        temp_path
    ]
    
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=60)
        chunk, _ = librosa.load(temp_path, sr=sample_rate, mono=True)
        os.unlink(temp_path)
        return chunk
    except subprocess.TimeoutExpired:
        os.unlink(temp_path)
        return None
    except:
        os.unlink(temp_path)
        return None


def load_audio_chunk_fallback(file_path, sample_rate, offset, duration):
    try:
        chunk = librosa.load(file_path, sr=sample_rate, mono=True, 
                            offset=offset, duration=duration)[0]
        return chunk
    except:
        return None


def load_audio_chunk(file_path, sample_rate, offset, duration):
    chunk = load_audio_chunk_ffmpeg(file_path, sample_rate, offset, duration)
    if chunk is None:
        chunk = load_audio_chunk_fallback(file_path, sample_rate, offset, duration)
    return chunk


def process_single_chunk(audio_chunk, highpass_cutoff, noise_reduction, 
                        silence_threshold, min_silence_duration, sample_rate,
                        scene='default'):
    silence_count = 0
    from .adaptive_processor import apply_highpass_filter
    
    try:
        if scene == 'cycling' or scene == 'cycling_bluetooth':
            from .cycling_audio_processor import apply_bandpass_filter, apply_voice_enhancement
            from .cycling_audio_processor import apply_dynamic_range_compression, apply_intelligibility_boost
            
            audio_chunk = apply_bandpass_filter(audio_chunk, sample_rate)
            audio_chunk = apply_voice_enhancement(audio_chunk, sample_rate)
            audio_chunk = apply_dynamic_range_compression(audio_chunk, sample_rate)
            audio_chunk = apply_intelligibility_boost(audio_chunk, sample_rate)
        elif highpass_cutoff > 0:
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
                del reduced_chunk
                gc.collect()
            except Exception:
                pass
        
        frame_length = int(sample_rate * 0.02)
        hop_length = int(sample_rate * 0.01)
        
        rms = librosa.feature.rms(y=audio_chunk, frame_length=frame_length, hop_length=hop_length)
        db = librosa.amplitude_to_db(rms, ref=1.0)
        
        silence_mask = db < silence_threshold
        silence_ratio = float(np.mean(silence_mask))
        
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
        
        silence_count = len(silence_segments)
        
        if silence_count > 0 or silence_ratio > 0.1:
            print(f"[Silence] dB范围: [{np.min(db):.1f}, {np.max(db):.1f}], 阈值={silence_threshold}, 静音比例={silence_ratio*100:.1f}%, 检测到{silence_count}个静音段")
        
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
        
        return audio_chunk, silence_count
    except Exception:
        return audio_chunk, 0


def _chunk_worker(args):
    i, audio_chunk, highpass_cutoff, noise_reduction, silence_threshold, \
        min_silence_duration, sample_rate, result_path, scene = args
    
    result, silence_count = process_single_chunk(audio_chunk, highpass_cutoff, noise_reduction,
                                     silence_threshold, min_silence_duration, sample_rate, scene)
    np.save(result_path, result)
    np.save(result_path + '_silence', np.int64(silence_count))


def process_chunk_with_timeout(i, audio_chunk, highpass_cutoff, noise_reduction, 
                               silence_threshold, min_silence_duration, sample_rate, result_path, scene='default'):
    args = (i, audio_chunk, highpass_cutoff, noise_reduction, silence_threshold, 
            min_silence_duration, sample_rate, result_path, scene)
    
    p = mp.Process(target=_chunk_worker, args=(args,))
    p.start()
    p.join(timeout=120)
    
    if p.is_alive():
        p.terminate()
        p.join()
        return False, audio_chunk
    
    if os.path.exists(result_path):
        try:
            result = np.load(result_path)
            os.unlink(result_path)
            return True, result
        except:
            os.unlink(result_path)
            return False, audio_chunk
    
    return False, audio_chunk


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
                         progress_callback=None, task_name="audio", scene='default'):
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
        "silence_segments_removed": 0
    }
    
    start_time = time.time()
    
    try:
        for i in range(num_chunks):
            chunk_start_time = time.time()
            
            offset = i * chunk_duration
            
            print(f"[{task_name}] 加载块 {i+1}/{num_chunks} (offset={offset:.1f}s)")
            
            audio_chunk = load_audio_chunk(file_path, sample_rate, offset, chunk_duration)
            if audio_chunk is None or len(audio_chunk) == 0:
                print(f"[{task_name}] 加载块 {i+1} 失败，跳过")
                stats["skipped_chunks"] += 1
                continue
            
            if progress_callback:
                progress_callback(int((i / num_chunks) * 100), 
                                f'正在处理块 {i+1}/{num_chunks}...')
            
            result_file = tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
            result_file.close()
            result_path = result_file.name
            
            success, result = process_chunk_with_timeout(
                i, audio_chunk, highpass_cutoff, noise_reduction,
                silence_threshold, min_silence_duration, sample_rate, result_path, scene
            )
            
            chunk_to_save = result if success else audio_chunk
            
            if success:
                stats["processed_chunks"] += 1
                if os.path.exists(result_path + '_silence'):
                    try:
                        silence_count = np.load(result_path + '_silence')
                        stats["silence_segments_removed"] += int(silence_count)
                        os.unlink(result_path + '_silence')
                    except:
                        pass
            else:
                stats["timeout_chunks"] += 1
                print(f"[{task_name}] 块 {i+1} 超时或失败，保留原始音频")
            
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_wav.close()
            temp_wav_path = temp_wav.name
            
            write_single_chunk_to_wav(chunk_to_save, sample_rate, temp_wav_path)
            current_batch.append(temp_wav_path)
            
            del audio_chunk, result, chunk_to_save
            gc.collect()
            
            if len(current_batch) >= BATCH_SIZE or (i == num_chunks - 1 and current_batch):
                merged_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                merged_temp.close()
                merged_path = merged_temp.name
                
                if merge_wav_files(current_batch, merged_path):
                    temp_wav_files.append(merged_path)
                    stats["temp_files_created"] += 1
                    
                    for wav_file in current_batch:
                        os.unlink(wav_file)
                else:
                    temp_wav_files.extend(current_batch)
                    stats["temp_files_created"] += len(current_batch)
                
                current_batch = []
                gc.collect()
            
            chunk_time = time.time() - chunk_start_time
            stats["chunk_times"].append(chunk_time)
            
            print(f"[{task_name}] 块 {i+1}/{num_chunks} 完成，耗时 {chunk_time:.2f}s")
        
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
        
        return np.array([]), {"error": str(e)}, None
    
    avg_time = np.mean(stats["chunk_times"]) if stats["chunk_times"] else 0
    max_time = np.max(stats["chunk_times"]) if stats["chunk_times"] else 0
    
    print(f"[{task_name}] 处理完成: 总块数={stats['total_chunks']}, 处理={stats['processed_chunks']}, "
          f"超时={stats['timeout_chunks']}, 跳过={stats['skipped_chunks']}, "
          f"临时文件={stats['temp_files_created']}, 总耗时={stats['total_time']:.2f}s, "
          f"平均每块={avg_time:.2f}s, 最长={max_time:.2f}s")
    
    return processed_audio, stats, temp_wav_path