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
    """判断是否为临时上传文件（带UUID前缀或task_前缀）"""
    import re
    uuid_pattern = r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}_'
    task_pattern = r'^task_\d+_'
    return bool(re.match(uuid_pattern, filename)) or bool(re.match(task_pattern, filename))

def cleanup_temp_files():
    """
    清理input和output目录中的所有临时文件（带UUID前缀的文件）
    不删除源文件（用户指定的输入文件）和已处理的输出文件
    """
    cleaned_count = 0
    cleaned_size = 0
    
    for directory in [UPLOAD_DIR, OUTPUT_DIR]:
        if not os.path.exists(directory):
            continue
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            if not is_temp_file(filename):
                continue
            
            try:
                file_size = os.path.getsize(file_path)
                os.remove(file_path)
                cleaned_count += 1
                cleaned_size += file_size
                print(f"清理临时文件: {filename}")
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
