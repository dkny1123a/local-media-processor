"""工具函数测试"""
import os
import tempfile
import shutil

def test_is_temp_file_uuid_prefix():
    """测试UUID前缀的临时文件识别"""
    from backend.utils import is_temp_file
    
    assert is_temp_file('a1b2c3d4-e5f6-7890-abcd-ef1234567890_test.mp3') is True
    assert is_temp_file('00000000-0000-0000-0000-000000000000_file.wav') is True

def test_is_temp_file_task_prefix():
    """测试task前缀的临时文件识别"""
    from backend.utils import is_temp_file
    
    assert is_temp_file('task_1234567890_test.mp3') is True
    assert is_temp_file('task_1783648186713_ts9kow2ky_filename.m4a') is True
    assert is_temp_file('task_0_test.mp3') is True

def test_is_temp_file_non_temp():
    """测试非临时文件的识别"""
    from backend.utils import is_temp_file
    
    assert is_temp_file('processed_file.mp3') is False
    assert is_temp_file('source_file.mp3') is False
    assert is_temp_file('20260623004001_processed.mp3') is False
    assert is_temp_file('26113秦皇湖_processed.mp3') is False

def test_is_temp_file_empty_and_edge_cases():
    """测试边界情况"""
    from backend.utils import is_temp_file
    
    assert is_temp_file('') is False
    assert is_temp_file('task_') is False
    assert is_temp_file('task') is False
    assert is_temp_file('abc123_test.mp3') is False

def test_cleanup_temp_files(test_dirs):
    """测试清理临时文件"""
    from backend.utils import cleanup_temp_files
    
    upload_dir, output_dir = test_dirs
    
    temp_files = [
        os.path.join(upload_dir, 'task_1234567890_temp1.mp3'),
        os.path.join(upload_dir, 'a1b2c3d4-e5f6-7890-abcd-ef1234567890_temp2.wav'),
        os.path.join(output_dir, 'task_1234567890_temp3_processed.mp3'),
        os.path.join(output_dir, 'a1b2c3d4-e5f6-7890-abcd-ef1234567890_temp4_processed.wav'),
    ]
    
    non_temp_files = [
        os.path.join(upload_dir, 'source_file.mp3'),
        os.path.join(output_dir, 'processed_file.mp3'),
        os.path.join(output_dir, '26113秦皇湖_processed.mp3'),
    ]
    
    for fpath in temp_files + non_temp_files:
        with open(fpath, 'w') as f:
            f.write('test content')
    
    cleaned_count, cleaned_size = cleanup_temp_files()
    
    assert cleaned_count == 4
    assert cleaned_size > 0
    
    for fpath in temp_files:
        assert os.path.exists(fpath) is False
    
    for fpath in non_temp_files:
        assert os.path.exists(fpath) is True

def test_get_file_stats(test_dirs):
    """测试文件统计"""
    from backend.utils import get_file_stats
    
    upload_dir, output_dir = test_dirs
    
    temp_files = [
        os.path.join(upload_dir, 'task_1234567890_temp1.mp3'),
        os.path.join(upload_dir, 'a1b2c3d4-e5f6-7890-abcd-ef1234567890_temp2.wav'),
    ]
    
    non_temp_files = [
        os.path.join(upload_dir, 'source_file.mp3'),
    ]
    
    for fpath in temp_files + non_temp_files:
        with open(fpath, 'w') as f:
            f.write('test content')
    
    stats = get_file_stats(upload_dir)
    
    assert stats["count"] == 3
    assert stats["size"] > 0
    assert stats["temp_count"] == 2
    assert stats["source_count"] == 1

def test_get_file_stats_empty_dir():
    """测试空目录统计"""
    from backend.utils import get_file_stats
    
    with tempfile.TemporaryDirectory() as tmpdir:
        stats = get_file_stats(tmpdir)
        assert stats["count"] == 0
        assert stats["size"] == 0
        assert stats["temp_count"] == 0
        assert stats["source_count"] == 0

def test_get_file_stats_nonexistent_dir():
    """测试不存在的目录统计"""
    from backend.utils import get_file_stats
    
    stats = get_file_stats('/nonexistent/directory/path')
    assert stats["count"] == 0
    assert stats["size"] == 0
    assert stats["temp_count"] == 0
    assert stats["source_count"] == 0