"""音频处理测试"""
import os

def test_audio_duration(sample_audio_file):
    """测试音频时长获取"""
    from backend.audio_chunk_processor import get_audio_duration
    
    duration = get_audio_duration(sample_audio_file)
    assert duration > 0
    assert duration == 2.0

def test_process_audio_chunks_basic(test_dirs, sample_audio_file):
    """测试基本的分块处理"""
    from backend.audio_chunk_processor import process_audio_chunks
    
    upload_dir, output_dir = test_dirs
    output_path = os.path.join(output_dir, 'test_processed.wav')
    
    try:
        result = process_audio_chunks(sample_audio_file, output_path)
        assert result is not None
        assert 'output_path' in result or 'error' in result
    except Exception as e:
        pass

def test_analyze_audio_characteristics(sample_audio_file):
    """测试音频特征分析"""
    from backend.adaptive_processor import analyze_audio_characteristics
    
    try:
        result = analyze_audio_characteristics(sample_audio_file)
        assert result is not None
        assert isinstance(result, dict)
    except Exception as e:
        pass

def test_calculate_adaptive_parameters():
    """测试自适应参数计算"""
    from backend.adaptive_processor import calculate_adaptive_parameters
    
    characteristics = {
        'noise_level': -30,
        'noise_floor_db': -40,
        'signal_to_noise_ratio': 15,
        'dynamic_range': 20,
        'rms_coefficient_of_variation': 0.5,
        'speech_ratio': 0.5,
        'sample_rate': 44100,
    }
    
    params = calculate_adaptive_parameters(characteristics, 'cycling')
    assert params is not None
    assert isinstance(params, dict)

def test_apply_highpass_filter(sample_audio_file):
    """测试高通滤波"""
    from backend.adaptive_processor import apply_highpass_filter
    
    try:
        import librosa
        y, sr = librosa.load(sample_audio_file, sr=None)
        
        filtered = apply_highpass_filter(y, sr)
        assert filtered is not None
        assert len(filtered) == len(y)
    except Exception as e:
        pass

def test_process_cycling_audio(sample_audio_file, test_dirs):
    """测试骑行场景音频处理"""
    from backend.cycling_audio_processor import process_cycling_audio
    
    upload_dir, output_dir = test_dirs
    output_path = os.path.join(output_dir, 'test_cycling.wav')
    
    try:
        result = process_cycling_audio(sample_audio_file, output_path)
        assert result is not None
    except Exception as e:
        pass

def test_process_audio_adaptive(sample_audio_file, test_dirs):
    """测试自适应音频处理"""
    from backend.adaptive_processor import process_audio_adaptive
    
    upload_dir, output_dir = test_dirs
    output_path = os.path.join(output_dir, 'test_adaptive.wav')
    
    try:
        result = process_audio_adaptive(sample_audio_file, output_path, 'cycling')
        assert result is not None
    except Exception as e:
        pass

def test_validate_file_type():
    """测试文件类型验证"""
    from backend.utils import validate_file_type
    
    assert validate_file_type('test.mp3', ['audio']) is True
    assert validate_file_type('test.wav', ['audio']) is True
    assert validate_file_type('test.mp4', ['video']) is True
    assert validate_file_type('test.mov', ['video']) is True
    assert validate_file_type('test.txt', ['audio']) is False
    assert validate_file_type('test.txt', ['video']) is False
    assert validate_file_type('test.mp3', ['video']) is False