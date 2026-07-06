import uuid
import os
import time
from datetime import datetime

UPLOAD_DIR = os.environ.get('UPLOAD_DIR', '/app/uploads')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '/app/output')

def validate_file_type(filename, allowed_types):
    ext = os.path.splitext(filename)[1].lower()
    audio_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.amr', '.3gp']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    if 'audio' in allowed_types and ext in audio_extensions:
        return True
    if 'video' in allowed_types and ext in video_extensions:
        return True
    return False

def generate_file_id():
    return str(uuid.uuid4())

def generate_unique_id():
    return str(uuid.uuid4())

def is_temp_file(filename):
    """判断是否为临时上传文件（带UUID前缀）"""
    # UUID格式: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    uuid_pattern = r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}_'
    import re
    return bool(re.match(uuid_pattern, filename))

def cleanup_temp_files(max_age_hours=24):
    """
    清理上传目录中的临时文件（带UUID前缀的文件）
    不删除源文件（用户指定的输入文件）和已处理的输出文件
    """
    cleaned_count = 0
    cleaned_size = 0
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    if not os.path.exists(UPLOAD_DIR):
        return cleaned_count, cleaned_size
    
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        # 只清理临时文件（带UUID前缀）
        if not is_temp_file(filename):
            continue
        
        # 检查文件年龄
        try:
            file_mtime = os.path.getmtime(file_path)
            file_age = current_time - file_mtime
            
            if file_age > max_age_seconds:
                file_size = os.path.getsize(file_path)
                os.remove(file_path)
                cleaned_count += 1
                cleaned_size += file_size
                print(f"清理临时文件: {filename} (年龄: {file_age/3600:.1f}小时)")
        except Exception as e:
            print(f"清理文件失败 {filename}: {str(e)}")
    
    return cleaned_count, cleaned_size

def get_file_stats(directory):
    """获取目录文件统计"""
    if not os.path.exists(directory):
        return {"count": 0, "size": 0, "temp_count": 0, "source_count": 0}
    
    total_count = 0
    total_size = 0
    temp_count = 0
    source_count = 0
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            total_count += 1
            total_size += os.path.getsize(file_path)
            if is_temp_file(filename):
                temp_count += 1
            else:
                source_count += 1
    
    return {
        "count": total_count,
        "size": total_size,
        "temp_count": temp_count,
        "source_count": source_count
    }
