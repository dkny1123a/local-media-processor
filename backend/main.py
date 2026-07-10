from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import os
import uuid
import time
import asyncio
import threading
import multiprocessing as mp
from .audio_processor import process_audio
from .video_processor import process_video, extract_audio_from_video, get_audio_info
from .utils import cleanup_temp_files, get_file_stats, is_temp_file
from .denoise_processor import process_denoise
from .adaptive_processor import process_audio_adaptive, analyze_audio_characteristics, calculate_adaptive_parameters, apply_highpass_filter
from .audio_chunk_processor import process_audio_chunks, load_audio_chunk, get_audio_duration
from .cycling_audio_processor import process_cycling_audio

MAX_FILE_SIZE = 4 * 1024 * 1024 * 1024

SUPPORTED_AUDIO_EXTS = ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.amr', '.3gp']
SUPPORTED_VIDEO_EXTS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
SUPPORTED_EXTS = SUPPORTED_AUDIO_EXTS + SUPPORTED_VIDEO_EXTS

app = FastAPI(
    title="本地多媒体处理器", 
    description="本地AI音频处理工具",
)

progress_states = {}
progress_lock = threading.Lock()
task_results = {}
task_results_lock = threading.Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/app/uploads")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)



def update_progress(task_id, status, message, percentage=0):
    with progress_lock:
        progress_states[task_id] = {
            'status': status,
            'message': message,
            'percentage': percentage,
            'timestamp': time.time()
        }

def get_progress(task_id):
    with progress_lock:
        return progress_states.get(task_id, None)

def clear_progress(task_id):
    with progress_lock:
        if task_id in progress_states:
            del progress_states[task_id]

def save_task_result(task_id, result):
    with task_results_lock:
        task_results[task_id] = result

def get_task_result(task_id):
    with task_results_lock:
        return task_results.get(task_id, None)

def clear_task_result(task_id):
    with task_results_lock:
        if task_id in task_results:
            del task_results[task_id]

@app.get("/api/media/progress/{task_id}")
async def get_progress_endpoint(task_id: str):
    progress = get_progress(task_id)
    if progress:
        return progress
    return {"status": "not_found", "message": "任务不存在或已完成"}

@app.get("/api/media/result/{task_id}")
async def get_result_endpoint(task_id: str):
    result = get_task_result(task_id)
    if result:
        return result
    return {"status": "pending", "message": "任务正在处理中"}

def resolve_host_path(path):
    if not path:
        return None
    path = os.path.expanduser(path)
    if path.startswith('/app'):
        return path
    if path.startswith('/hostroot'):
        path = path.replace('/hostroot', '', 1)
    return path

def get_output_path(output_path, original_filename, suffix="processed"):
    if not output_path:
        base_name = os.path.splitext(original_filename)[0] if original_filename else f"file_{uuid.uuid4().hex[:8]}"
        filename = f"{base_name}_{suffix}.mp3"
        return os.path.join(OUTPUT_DIR, filename)
    elif os.path.isdir(output_path):
        base_name = os.path.splitext(original_filename)[0] if original_filename else f"file_{uuid.uuid4().hex[:8]}"
        filename = f"{base_name}_{suffix}.mp3"
        return os.path.join(output_path, filename)
    else:
        return output_path

def convert_to_wav(input_path):
    import subprocess
    import tempfile
    
    ext = os.path.splitext(input_path)[1].lower()
    
    amr_formats = ['.amr', '.3gp']
    m4a_formats = ['.m4a']
    
    if ext in amr_formats or ext in m4a_formats:
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_path = temp_wav.name
        temp_wav.close()
        
        command = [
            'ffmpeg',
            '-i', input_path,
            '-ac', '1',
            '-ar', '44100',
            '-y',
            temp_path
        ]
        
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=120)
            return temp_path, True
        except subprocess.CalledProcessError as e:
            print(f"格式转换失败: {e.stderr.decode()}")
            return input_path, False
        except subprocess.TimeoutExpired:
            print(f"格式转换超时")
            os.unlink(temp_path)
            return input_path, False
    
    return input_path, False

def detect_file_type(filename):
    ext = os.path.splitext(filename)[1].lower()
    
    if ext in SUPPORTED_AUDIO_EXTS:
        return 'audio'
    elif ext in SUPPORTED_VIDEO_EXTS:
        return 'video'
    else:
        return None

def get_audio_duration(file_path):
    import subprocess
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
        pass
    
    try:
        import librosa
        return librosa.get_duration(path=file_path)
    except:
        return 0

def load_audio_chunk_ffmpeg(file_path, sample_rate, offset, duration):
    import subprocess
    import numpy as np
    import tempfile
    
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
        
        import librosa
        chunk, _ = librosa.load(temp_path, sr=sample_rate, mono=True)
        
        os.unlink(temp_path)
        return chunk
    except subprocess.TimeoutExpired:
        os.unlink(temp_path)
        return None
    except:
        os.unlink(temp_path)
        return None

def load_audio_chunks(file_path, sample_rate, chunk_duration=60):
    chunk_size = int(sample_rate * chunk_duration)
    
    total_duration = get_audio_duration(file_path)
    if total_duration <= 0:
        import librosa
        total_duration = librosa.get_duration(path=file_path)
    
    total_samples = int(total_duration * sample_rate)
    num_chunks = (total_samples + chunk_size - 1) // chunk_size
    
    def chunk_generator():
        for i in range(num_chunks):
            offset = i * chunk_duration
            chunk = load_audio_chunk_ffmpeg(file_path, sample_rate, offset, chunk_duration)
            if chunk is None:
                try:
                    import librosa
                    chunk = librosa.load(file_path, sr=sample_rate, mono=True, 
                                        offset=offset, duration=chunk_duration)[0]
                except:
                    chunk = None
            if chunk is not None:
                yield chunk
                del chunk
    
    return chunk_generator, num_chunks, total_samples, sample_rate


def cleanup_uploaded_file(task_id, input_path):
    """处理完成后清理上传目录中的临时文件，避免空间占用。
    仅删除通过上传方式创建的临时文件（位于UPLOAD_DIR且以task_id开头），
    不会删除用户通过input_path指定的原始文件。"""
    try:
        if not input_path or not os.path.exists(input_path):
            return
        # 仅清理位于上传目录且以task_id开头的文件
        upload_dir = os.path.abspath(UPLOAD_DIR)
        abs_input = os.path.abspath(input_path)
        if abs_input.startswith(upload_dir):
            basename = os.path.basename(input_path)
            if basename.startswith(f"{task_id}_"):
                os.unlink(input_path)
                print(f"[Task {task_id}] 已清理上传临时文件: {basename}")
    except Exception as e:
        print(f"[Task {task_id}] 清理上传临时文件失败: {e}")


def process_audio_background(
    task_id, input_path, output_path, original_filename,
    auto_detect, noise_reduction, silence_threshold, min_silence_duration,
    max_volume, stationary_noise, scene
):
    import librosa
    import numpy as np
    import tempfile
    import subprocess
    import gc

    print(f"[Task {task_id}] 开始后台处理: {original_filename}")
    
    try:
        print(f"[Task {task_id}] 转换格式: {input_path}")
        update_progress(task_id, 'converting', '正在转换音频格式...', 10)
        converted_path, was_converted = convert_to_wav(input_path)
        
        print(f"[Task {task_id}] 获取音频信息")
        update_progress(task_id, 'loading', '正在获取音频信息...', 15)
        
        try:
            total_duration = librosa.get_duration(path=converted_path)
            sr_result = librosa.get_samplerate(converted_path)
        except:
            total_duration = 300
            sr_result = 44100
        
        sample_rate = sr_result
        original_duration = total_duration
        print(f"[Task {task_id}] 音频时长: {original_duration:.2f}秒, 采样率: {sample_rate}")
        
        chunk_duration = 60
        chunk_size = int(sample_rate * chunk_duration)
        num_chunks = int(np.ceil(total_duration / chunk_duration))
        
        update_progress(task_id, 'analyzing', '正在分析音频特征（噪声等级、动态范围）...', 20)
        
        chunk_generator, _, total_samples, _ = load_audio_chunks(converted_path, sample_rate, chunk_duration)
        
        analysis = None
        for i, chunk in enumerate(chunk_generator()):
            if i == 0:
                analysis = analyze_audio_characteristics(chunk, sample_rate)
                print(f"[Task {task_id}] 分析完成: noise_level={analysis['noise_level']}, snr={analysis['signal_to_noise_ratio']:.1f}")
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
            print(f"[Task {task_id}] 自适应参数: nr={noise_reduction}, hp={highpass_cutoff}, st={silence_threshold}")
        else:
            highpass_cutoff = 100.0 if scene == 'cycling' else 0.0
            target_db = -3.0 if scene in ['bluetooth', 'cycling'] else -1.0
        
        converted_path_again, was_converted_again = convert_to_wav(input_path)
        
        temp_wav_path = None
        cycling_stats = None
        
        update_progress(task_id, 'processing', '骑行+蓝牙场景专用处理...', 25)
        
        def progress_callback(pct, msg, status=None):
            update_progress(task_id, 'processing', msg, 25 + int(pct * 0.5))
        
        try:
            import librosa
            audio_data, sr = librosa.load(converted_path_again, sr=sample_rate, mono=True)
            
            processed_audio, cycling_stats, temp_wav_path = process_cycling_audio(
                audio_data, sample_rate,
                noise_reduction=noise_reduction,
                silence_threshold=silence_threshold,
                min_silence_duration=min_silence_duration,
                highpass_cutoff=highpass_cutoff,
                max_volume=max_volume,
                target_db=target_db,
                progress_callback=progress_callback
            )
            
            del audio_data
            gc.collect()
            
            sample_rate = 16000
        except Exception as e:
            print(f"[Task {task_id}] 统一处理失败，回退到分块处理: {e}")
            
            def fallback_progress_callback(msg, pct):
                update_progress(task_id, 'processing', msg, 25 + int(pct * 0.5))
            
            processed_audio, stats, temp_wav_path = process_audio_chunks(
                converted_path_again, sample_rate, chunk_duration,
                highpass_cutoff, noise_reduction, silence_threshold, min_silence_duration,
                progress_callback=fallback_progress_callback, task_name=f"Task {task_id}", scene='cycling_bluetooth'
            )
        
        if cycling_stats is None:
            update_progress(task_id, 'processing', '蓝牙优化（降采样至16kHz）...', 85)
            
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
            except subprocess.TimeoutExpired:
                print(f"[Task {task_id}] 降采样超时")
                os.unlink(optimized_path)
            except:
                os.unlink(optimized_path)
            
            if was_converted_again and os.path.exists(converted_path_again):
                os.unlink(converted_path_again)
            
            if max_volume and temp_wav_path:
                update_progress(task_id, 'processing', '正在调整音量...', 75)
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
                    print(f"[Task {task_id}] 音量调整超时")
                    os.unlink(normalized_path)
                except:
                    os.unlink(normalized_path)
        else:
            if was_converted_again and os.path.exists(converted_path_again):
                os.unlink(converted_path_again)
        
        silence_segments_removed = cycling_stats.get('silence_segments_removed', 0) if cycling_stats else 0
        
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
                'silence_segments_removed': silence_segments_removed
            }
        }
        
        if not adaptive_result['success']:
            print(f"[Task {task_id}] 自适应处理失败: {adaptive_result['message']}")
            result = {
                "success": False,
                "message": adaptive_result['message'],
                "file_type": "audio"
            }
            save_task_result(task_id, result)
            clear_progress(task_id)
            cleanup_uploaded_file(task_id, input_path)
            return
        
        update_progress(task_id, 'encoding', '正在编码为MP3格式...', 90)
        
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        if temp_wav_path and os.path.exists(temp_wav_path):
            command = [
                'ffmpeg',
                '-i', temp_wav_path,
                '-ac', '1',
                '-ar', str(sample_rate),
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
                '-ar', str(sample_rate),
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
        
        print(f"[Task {task_id}] 处理完成: {output_path}")
        save_task_result(task_id, result)
        update_progress(task_id, 'complete', '处理完成', 100)
        clear_progress(task_id)
        cleanup_uploaded_file(task_id, input_path)

    except Exception as e:
        print(f"[Task {task_id}] 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        result = {
            "success": False,
            "message": f"音频处理失败: {str(e)}",
            "file_type": "audio"
        }
        save_task_result(task_id, result)
        clear_progress(task_id)
        cleanup_uploaded_file(task_id, input_path)


@app.post("/api/media/process")
async def process_media_endpoint(
    file: UploadFile = File(None),
    input_path: str = Form(None),
    output_path: str = Form(None),
    silence_threshold: float = Form(-40.0),
    min_silence_duration: float = Form(0.5),
    max_volume: bool = Form(True),
    noise_reduction: float = Form(0.0),
    stationary_noise: bool = Form(False),
    auto_detect: bool = Form(True),
    scene: str = Form("bluetooth"),
    task_id: str = Form(None)
):
    if not task_id:
        task_id = str(uuid.uuid4())
    
    update_progress(task_id, 'started', '任务已开始', 0)
    
    original_filename = None
    
    if file:
        original_filename = file.filename
        input_path = os.path.join(UPLOAD_DIR, f"{task_id}_{file.filename}")
        with open(input_path, "wb") as f:
            f.write(await file.read())
    else:
        if input_path and input_path.startswith('/'):
            input_path = resolve_host_path(input_path)
        if input_path:
            original_filename = os.path.basename(input_path)
    
    update_progress(task_id, 'uploaded', '文件上传完成', 10)
    
    if not original_filename:
        clear_progress(task_id)
        return {"success": False, "message": "无法获取文件名"}
    
    file_type = detect_file_type(original_filename)
    
    if not file_type:
        clear_progress(task_id)
        return {"success": False, "message": "不支持的文件格式，请上传音频或视频文件"}
    
    if output_path and output_path.startswith('/') and not output_path.startswith('/app'):
        output_path = resolve_host_path(output_path)
    
    if file_type == 'audio':
        output_path = get_output_path(output_path, original_filename, "processed")
        
        thread = threading.Thread(
            target=process_audio_background,
            args=(
                task_id, input_path, output_path, original_filename,
                auto_detect, noise_reduction, silence_threshold, min_silence_duration,
                max_volume, stationary_noise, scene
            ),
            daemon=True
        )
        thread.start()
        
        return {"success": True, "message": "任务已提交", "task_id": task_id}
    
    else:
        base_name = os.path.splitext(original_filename)[0]
        audio_output_path = os.path.join(OUTPUT_DIR, f"{base_name}_audio.mp3")
        processed_audio_path = os.path.join(OUTPUT_DIR, f"{base_name}_processed.mp3")
        
        if output_path:
            if os.path.isdir(output_path):
                audio_output_path = os.path.join(output_path, f"{base_name}_audio.mp3")
                processed_audio_path = os.path.join(output_path, f"{base_name}_processed.mp3")
            else:
                processed_audio_path = output_path
        
        def process_video_background():
            print(f"[Task {task_id}] 开始视频处理: {original_filename}")
            try:
                result = process_video(
                    input_path, 
                    audio_output_path, 
                    processed_audio_path, 
                    extract_audio=True, 
                    do_process_audio=True,
                    silence_threshold=silence_threshold,
                    min_silence_duration=min_silence_duration,
                    max_volume=max_volume,
                    noise_reduction=noise_reduction,
                    stationary_noise=stationary_noise,
                    auto_detect=auto_detect,
                    scene=scene,
                    progress_callback=lambda pct, msg, status=None: update_progress(
                        task_id, status or 'processing', msg, int(pct * 100)
                    )
                )
                
                if result["success"]:
                    audio_info = result.get("audio_info", {})
                    processed_info = result.get("processed_info", {})
                    original_duration = audio_info.get("duration", 0)
                    processed_duration = processed_info.get("duration", 0)
                    duration_reduction = ((original_duration - processed_duration) / original_duration * 100) if original_duration > 0 else 0
                    
                    print(f"[Task {task_id}] 构建最终结果: original_duration={original_duration}, processed_duration={processed_duration}")
                    
                    final_result = {
                        "success": True,
                        "message": result["message"],
                        "file_type": "video",
                        "processed_audio_file": os.path.basename(processed_audio_path),
                        "output_path": processed_audio_path,
                        "audio_info": audio_info,
                        "processed_info": processed_info,
                        "stats": {
                            "original_duration": round(original_duration, 2),
                            "processed_duration": round(processed_duration, 2),
                            "silence_segments_removed": 0,
                            "duration_reduction_percent": round(duration_reduction, 2),
                            "noise_reduction": result.get("applied_params", {}).get("noise_reduction", noise_reduction),
                            "sample_rate": processed_info.get("sample_rate", 44100)
                        },
                        "analysis": result.get("analysis", {}),
                        "applied_params": result.get("applied_params", {})
                    }
                else:
                    print(f"[Task {task_id}] 视频处理失败: {result['message']}")
                    final_result = {
                        "success": False,
                        "message": result["message"],
                        "file_type": "video"
                    }
                
                print(f"[Task {task_id}] 保存任务结果")
                save_task_result(task_id, final_result)
                print(f"[Task {task_id}] 清理进度状态")
                clear_progress(task_id)
                print(f"[Task {task_id}] 清理上传临时文件")
                cleanup_uploaded_file(task_id, input_path)

            except Exception as e:
                print(f"[Task {task_id}] 视频处理失败: {str(e)}")
                import traceback
                traceback.print_exc()
                save_task_result(task_id, {
                    "success": False,
                    "message": f"视频处理失败: {str(e)}",
                    "file_type": "video"
                })
                clear_progress(task_id)
                cleanup_uploaded_file(task_id, input_path)
        
        thread = threading.Thread(
            target=process_video_background,
            daemon=True
        )
        thread.start()
        
        return {"success": True, "message": "任务已提交", "task_id": task_id}

@app.post("/api/audio/waveform")
async def get_audio_waveform(
    request: Request,
    file: UploadFile = File(None),
    input_path: str = Form(None)
):
    import numpy as np
    import soundfile as sf
    from io import BytesIO
    import base64
    
    file_id = str(uuid.uuid4())
    
    if file:
        temp_path = os.path.join(UPLOAD_DIR, f"{file_id}_audio.wav")
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        input_path = temp_path
    elif not input_path:
        try:
            body = await request.body()
            params = {}
            for pair in body.decode().split('&'):
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    params[key] = value
            input_path = params.get('input_path')
        except:
            pass
    
    if input_path and input_path.startswith('/') and not input_path.startswith('/app'):
        input_path = resolve_host_path(input_path)
    
    if not input_path or not os.path.exists(input_path):
        return {"error": f"文件不存在: {input_path}"}
    
    try:
        data, sr = sf.read(input_path)
        
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        samples_per_bar = max(1, len(data) // 200)
        waveform_data = []
        
        for i in range(0, len(data), samples_per_bar):
            segment = data[i:i + samples_per_bar]
            waveform_data.append(float(np.abs(segment).mean()))
        
        max_val = max(waveform_data) if waveform_data else 1.0
        waveform_normalized = [v / max_val for v in waveform_data]
        
        return {
            "success": True,
            "waveform": waveform_normalized,
            "sample_rate": sr,
            "duration": len(data) / sr,
            "channels": 1,
            "samples": len(data),
            "filename": os.path.basename(input_path)
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        if file and os.path.exists(input_path):
            os.remove(input_path)

@app.get("/api/files/download/{filename}")
@app.head("/api/files/download/{filename}")
async def download_file(filename: str):
    import urllib.parse
    decoded_filename = urllib.parse.unquote(filename)
    file_path = os.path.join(OUTPUT_DIR, decoded_filename)
    
    if not os.path.exists(file_path):
        return {"error": "文件不存在"}
    
    ext = os.path.splitext(file_path)[1].lower()
    media_types = {
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
        '.flac': 'audio/flac',
        '.ogg': 'audio/ogg',
        '.aac': 'audio/aac',
        '.m4a': 'audio/mp4',
        '.amr': 'audio/amr',
        '.3gp': 'audio/3gpp'
    }
    
    return FileResponse(file_path, media_type=media_types.get(ext, "application/octet-stream"), filename=decoded_filename)

@app.api_route("/api/audio/preview", methods=["GET", "HEAD"])
async def preview_audio(path: str, request: Request):
    import urllib.parse
    import re
    from starlette.responses import Response
    
    decoded_path = urllib.parse.unquote(path)
    
    resolved_path = resolve_host_path(decoded_path)
    
    if resolved_path and os.path.exists(resolved_path):
        file_path = resolved_path
    else:
        filename = os.path.basename(decoded_path)
        file_path = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(file_path):
        return {"error": f"文件不存在: {file_path}"}
    
    ext = os.path.splitext(file_path)[1].lower()
    media_types = {
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
        '.flac': 'audio/flac',
        '.ogg': 'audio/ogg',
        '.aac': 'audio/aac',
        '.m4a': 'audio/mp4',
        '.amr': 'audio/amr',
        '.3gp': 'audio/3gpp'
    }
    
    file_size = os.path.getsize(file_path)
    range_header = request.headers.get('range')
    
    if range_header:
        range_match = re.match(r'bytes=(\d+)-(\d*)', range_header)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2)) if range_match.group(2) else file_size - 1
            
            if start >= file_size:
                return JSONResponse({"error": "Range not satisfiable"}, status_code=416)
            
            end = min(end, file_size - 1)
            chunk_size = end - start + 1
            
            with open(file_path, 'rb') as f:
                f.seek(start)
                content = f.read(chunk_size)
            
            headers = {
                'Content-Range': f'bytes {start}-{end}/{file_size}',
                'Content-Length': str(chunk_size),
                'Accept-Ranges': 'bytes',
            }
            return Response(content, status_code=206, headers=headers, media_type=media_types.get(ext, "audio/mpeg"))
    
    return FileResponse(file_path, media_type=media_types.get(ext, "audio/mpeg"))

@app.get("/api/files/list")
async def list_files():
    files = []
    for f in os.listdir(OUTPUT_DIR):
        full_path = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(full_path):
            ext = os.path.splitext(f)[1].lower()
            if ext not in SUPPORTED_EXTS:
                continue
            
            duration = 0
            if ext in SUPPORTED_AUDIO_EXTS:
                duration = get_audio_duration(full_path)
            
            files.append({
                "name": f,
                "mtime": os.path.getmtime(full_path),
                "size": os.path.getsize(full_path),
                "duration": round(duration, 2) if duration > 0 else 0
            })
    files.sort(key=lambda x: x["mtime"], reverse=True)
    return {"files": files}

@app.get("/api/files/browse")
async def browse_directory(path: str = "/"):
    try:
        path = os.path.abspath(path)
        
        if not os.path.exists(path):
            return {"error": "路径不存在", "directories": [], "files": []}
        
        directories = []
        files = []
        
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)
            if os.path.isdir(full_path):
                directories.append({
                    "name": entry,
                    "path": full_path,
                    "type": "directory"
                })
            elif os.path.isfile(full_path):
                ext = os.path.splitext(entry)[1].lower()
                if ext in SUPPORTED_EXTS:
                    files.append({
                        "name": entry,
                        "path": full_path,
                        "type": "file"
                    })
        
        return {
            "current_path": path,
            "parent_path": os.path.dirname(path) if path != "/" else None,
            "directories": sorted(directories, key=lambda x: x["name"]),
            "files": sorted(files, key=lambda x: x["name"])
        }
    except Exception as e:
        return {"error": str(e), "directories": [], "files": []}

@app.get("/api/files/recent")
async def get_recent_files():
    recent = []
    try:
        for f in os.listdir(UPLOAD_DIR):
            full_path = os.path.join(UPLOAD_DIR, f)
            if os.path.isfile(full_path):
                recent.append({
                    "name": f,
                    "path": full_path
                })
        return {"files": recent}
    except Exception as e:
        return {"files": []}

@app.delete("/api/files/delete/{filename}")
async def delete_file_delete(filename: str):
    return await do_delete_file(filename)

@app.post("/api/files/delete")
async def delete_file_post(filename: str = Form(...)):
    return await do_delete_file(filename)

@app.post("/api/files/delete_batch")
async def delete_files_batch(filenames: str = Form(...)):
    import urllib.parse
    filenames_list = filenames.split(',')
    deleted_count = 0
    deleted_files = []
    
    for filename in filenames_list:
        filename = filename.strip()
        if not filename:
            continue
        decoded_filename = urllib.parse.unquote(filename)
        
        file_path_output = os.path.join(OUTPUT_DIR, decoded_filename)
        file_path_upload = os.path.join(UPLOAD_DIR, decoded_filename)
        
        if os.path.exists(file_path_output):
            os.remove(file_path_output)
            deleted_count += 1
            deleted_files.append(filename)
        elif os.path.exists(file_path_upload):
            os.remove(file_path_upload)
            deleted_count += 1
            deleted_files.append(filename)
    
    return {
        "success": True,
        "message": f"已删除 {deleted_count} 个文件",
        "deleted_count": deleted_count,
        "deleted_files": deleted_files
    }

async def do_delete_file(filename: str):
    import urllib.parse
    decoded_filename = urllib.parse.unquote(filename)
    
    file_path_output = os.path.join(OUTPUT_DIR, decoded_filename)
    file_path_upload = os.path.join(UPLOAD_DIR, decoded_filename)
    
    if os.path.exists(file_path_output):
        os.remove(file_path_output)
        return {"success": True, "message": f"文件 {filename} 已删除"}
    elif os.path.exists(file_path_upload):
        os.remove(file_path_upload)
        return {"success": True, "message": f"文件 {filename} 已删除"}
    
    return {"success": False, "message": f"文件 {filename} 不存在"}

try:
    from .async_api import async_router
    from .metrics import metrics_router
    app.include_router(async_router)
    app.include_router(metrics_router)
    print("INFO: 异步 API 和监控路由已注册")
except Exception as e:
    print(f"WARNING: 无法注册异步路由: {e}")

@app.get("/api/files/stats")
async def get_files_stats():
    output_stats = get_file_stats(OUTPUT_DIR)
    
    return {
        "output_dir": {
            "path": OUTPUT_DIR,
            "total_files": output_stats["count"],
            "total_size_mb": round(output_stats["size"] / (1024*1024), 2),
            "processed_files": output_stats["count"]
        }
    }

@app.post("/api/files/cleanup")
async def manual_cleanup():
    cleaned_count, cleaned_size = cleanup_temp_files()
    
    return {
        "success": True,
        "message": f"已清理 {cleaned_count} 个临时文件",
        "cleaned_count": cleaned_count,
        "cleaned_size_mb": round(cleaned_size / (1024*1024), 2)
    }

@app.post("/api/audio/concatenate")
async def concatenate_audio(
    filenames: str = Form(...)
):
    import subprocess
    import tempfile
    import json
    
    try:
        filename_list = json.loads(filenames)
        
        if len(filename_list) < 2:
            return {"success": False, "message": "至少需要选择2个文件进行拼接"}
        
        input_files = []
        for filename in filename_list:
            file_path = os.path.join(OUTPUT_DIR, filename)
            if not os.path.exists(file_path):
                return {"success": False, "message": f"文件不存在: {filename}"}
            input_files.append(file_path)
        
        timestamp = int(time.time())
        output_filename = f"concatenated_{timestamp}.mp3"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
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
            output_path
        ]
        
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=600)
            os.unlink(list_path)
        except subprocess.TimeoutExpired:
            os.unlink(list_path)
        except subprocess.CalledProcessError as e:
            os.unlink(list_path)
            
            temp_wav_files = []
            try:
                for i, file_path in enumerate(input_files):
                    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    temp_wav_path = temp_wav.name
                    temp_wav.close()
                    
                    convert_cmd = [
                        'ffmpeg',
                        '-i', file_path,
                        '-ac', '1',
                        '-ar', '44100',
                        '-y',
                        '-loglevel', 'quiet',
                        temp_wav_path
                    ]
                    subprocess.run(convert_cmd, check=True, capture_output=True, timeout=120)
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
                    '-ac', '1',
                    '-ar', '44100',
                    '-c:a', 'libmp3lame',
                    '-q:a', '2',
                    '-y',
                    '-loglevel', 'quiet',
                    output_path
                ]
                subprocess.run(concat_cmd, check=True, capture_output=True, timeout=600)
                os.unlink(concat_list_path)
                
                for wav_file in temp_wav_files:
                    if os.path.exists(wav_file):
                        os.unlink(wav_file)
            except Exception as convert_e:
                for wav_file in temp_wav_files:
                    if os.path.exists(wav_file):
                        os.unlink(wav_file)
                return {"success": False, "message": f"拼接失败: {str(convert_e)}"}
        
        total_duration = 0
        for file_path in input_files:
            try:
                result = subprocess.run(
                    ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', 
                     '-of', 'csv=p=0', file_path],
                    capture_output=True, text=True, check=True, timeout=30
                )
                total_duration += float(result.stdout.strip())
            except subprocess.TimeoutExpired:
                pass
            except:
                pass
        
        return {
            "success": True,
            "message": f"成功拼接 {len(filename_list)} 个音频文件",
            "output_file": output_filename,
            "output_path": output_path,
            "total_duration": round(total_duration, 2),
            "files_count": len(filename_list)
        }
    
    except json.JSONDecodeError:
        return {"success": False, "message": "文件名列表格式错误"}
    except Exception as e:
        return {"success": False, "message": f"拼接失败: {str(e)}"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)