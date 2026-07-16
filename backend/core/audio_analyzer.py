import numpy as np
import librosa


def analyze_audio_spectrum(audio_data: np.ndarray, sample_rate: int) -> dict:
    frame_length = int(sample_rate * 0.05)
    hop_length = int(sample_rate * 0.025)
    
    rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=1.0)
    
    spectral_flatness = librosa.feature.spectral_flatness(y=audio_data, n_fft=frame_length, hop_length=hop_length)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate, n_fft=frame_length, hop_length=hop_length)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate, n_fft=frame_length, hop_length=hop_length)[0]
    
    noise_floor_db = float(np.percentile(rms_db, 5))
    signal_peak_db = float(np.percentile(rms_db, 95))
    dynamic_range = signal_peak_db - noise_floor_db
    
    avg_flatness = float(np.mean(spectral_flatness))
    avg_centroid = float(np.mean(spectral_centroid))
    avg_bandwidth = float(np.mean(spectral_bandwidth))
    
    noise_score = 0.0
    if noise_floor_db > -55:
        noise_score += 0.6
    elif noise_floor_db > -65:
        noise_score += 0.3
    
    if avg_flatness > 0.3:
        noise_score += 0.4
    elif avg_flatness > 0.15:
        noise_score += 0.2
    
    if dynamic_range < 15:
        noise_score += 0.4
    elif dynamic_range < 25:
        noise_score += 0.2
    
    if avg_centroid < 1000:
        noise_score += 0.3
    
    if noise_score >= 0.8:
        noise_level = 'high'
    elif noise_score >= 0.5:
        noise_level = 'medium'
    else:
        noise_level = 'low'
    
    if dynamic_range > 25:
        signal_quality = 'good'
    elif dynamic_range > 12:
        signal_quality = 'medium'
    else:
        signal_quality = 'poor'
    
    return {
        'noise_floor_db': noise_floor_db,
        'signal_peak_db': signal_peak_db,
        'dynamic_range': dynamic_range,
        'spectral_flatness': avg_flatness,
        'spectral_centroid': avg_centroid,
        'spectral_bandwidth': avg_bandwidth,
        'noise_score': noise_score,
        'noise_level': noise_level,
        'signal_quality': signal_quality,
        'rms_mean': float(np.mean(rms)),
        'rms_std': float(np.std(rms)),
        'sample_rate': sample_rate,
        'duration': float(len(audio_data) / sample_rate),
    }