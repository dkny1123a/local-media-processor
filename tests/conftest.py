import pytest
import os
import tempfile
import shutil
from fastapi.testclient import TestClient

@pytest.fixture(scope="session")
def test_client():
    """FastAPI test client"""
    os.environ['UPLOAD_DIR'] = '/tmp/test_uploads'
    os.environ['OUTPUT_DIR'] = '/tmp/test_output'
    
    from backend.main import app
    return TestClient(app)

@pytest.fixture(scope="session")
def test_dirs():
    """Create test directories"""
    upload_dir = '/tmp/test_uploads'
    output_dir = '/tmp/test_output'
    
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    yield upload_dir, output_dir
    
    shutil.rmtree(upload_dir, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)

@pytest.fixture
def sample_audio_file():
    """Create a small test audio file"""
    import wave
    import struct
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        filename = f.name
    
    wav_file = wave.open(filename, 'w')
    wav_file.setparams((1, 2, 44100, 0, 'NONE', 'not compressed'))
    
    for i in range(44100 * 2):
        value = struct.pack('h', int(32767 * (i % 1000) / 1000))
        wav_file.writeframes(value)
    
    wav_file.close()
    
    yield filename
    
    os.remove(filename)

@pytest.fixture
def test_files(test_dirs):
    """Create test files in upload and output directories"""
    upload_dir, output_dir = test_dirs
    
    test_files = {
        'temp_upload': os.path.join(upload_dir, 'task_1234567890_test_temp.mp3'),
        'source_upload': os.path.join(upload_dir, 'source_file.mp3'),
        'temp_output': os.path.join(output_dir, 'task_1234567890_test_temp_processed.mp3'),
        'source_output': os.path.join(output_dir, 'processed_file.mp3'),
        'uuid_prefix': os.path.join(upload_dir, 'a1b2c3d4-e5f6-7890-abcd-ef1234567890_test.mp3'),
    }
    
    for fpath in test_files.values():
        with open(fpath, 'w') as f:
            f.write('test content')
    
    yield test_files
    
    for fpath in test_files.values():
        if os.path.exists(fpath):
            os.remove(fpath)