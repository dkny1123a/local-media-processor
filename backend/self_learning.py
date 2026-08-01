import os
import json
import numpy as np
import hashlib
from datetime import datetime
from typing import Dict, List, Optional

LEARNING_DB_PATH = "/tmp/audio_processing_history.json"
MAX_HISTORY = 100


class ProcessingRecord:
    def __init__(self, file_hash: str, features: Dict, parameters: Dict, result: Dict):
        self.file_hash = file_hash
        self.features = features
        self.parameters = parameters
        self.result = result
        self.timestamp = datetime.now().isoformat()

    def to_dict(self):
        return {
            'file_hash': self.file_hash,
            'features': self.features,
            'parameters': self.parameters,
            'result': self.result,
            'timestamp': self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict):
        record = cls(
            file_hash=data['file_hash'],
            features=data['features'],
            parameters=data['parameters'],
            result=data['result']
        )
        record.timestamp = data.get('timestamp', datetime.now().isoformat())
        return record


def compute_file_hash(file_path: str) -> str:
    h = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def load_history() -> List[ProcessingRecord]:
    try:
        if os.path.exists(LEARNING_DB_PATH):
            with open(LEARNING_DB_PATH, 'r') as f:
                data = json.load(f)
            return [ProcessingRecord.from_dict(item) for item in data]
    except Exception:
        pass
    return []


def save_history(records: List[ProcessingRecord]):
    trimmed = records[-MAX_HISTORY:]
    data = [r.to_dict() for r in trimmed]
    with open(LEARNING_DB_PATH, 'w') as f:
        json.dump(data, f, indent=2)


def compute_feature_distance(f1: Dict, f2: Dict) -> float:
    features = ['noise_floor_db', 'signal_peak_db', 'dynamic_range', 'spectral_flatness', 'spectral_centroid']
    weights = [0.3, 0.2, 0.25, 0.1, 0.15]
    
    distance = 0.0
    for feat, weight in zip(features, weights):
        v1 = f1.get(feat, 0)
        v2 = f2.get(feat, 0)
        normalized_diff = abs(v1 - v2)
        
        if feat == 'noise_floor_db':
            normalized_diff /= 60
        elif feat == 'signal_peak_db':
            normalized_diff /= 60
        elif feat == 'dynamic_range':
            normalized_diff /= 40
        elif feat == 'spectral_flatness':
            normalized_diff /= 1.0
        elif feat == 'spectral_centroid':
            normalized_diff /= 10000
        
        distance += weight * normalized_diff
    
    return distance


def find_similar_records(features: Dict, history: List[ProcessingRecord], top_n: int = 3) -> List[ProcessingRecord]:
    if not history:
        return []
    
    scored = [(compute_feature_distance(features, r.features), r) for r in history]
    scored.sort(key=lambda x: x[0])
    
    return [r for _, r in scored[:top_n]]


def analyze_audio_features(audio_data: np.ndarray, sample_rate: int, sample_duration: int = 30) -> Dict:
    try:
        import librosa
        
        if len(audio_data) / sample_rate > sample_duration:
            sample_size = int(sample_duration * sample_rate)
            num_samples = min(3, int(len(audio_data) / sample_size))
            samples = []
            for i in range(num_samples):
                start = int(i * len(audio_data) / num_samples)
                samples.append(audio_data[start:start + sample_size])
            audio_data = np.concatenate(samples)
        
        frame_length = int(sample_rate * 0.05)
        hop_length = int(sample_rate * 0.025)
        
        rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=1.0)
        
        spectral_flatness = librosa.feature.spectral_flatness(y=audio_data, n_fft=frame_length, hop_length=hop_length)[0]
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate, n_fft=frame_length, hop_length=hop_length)[0]
        
        result = {
            'noise_floor_db': float(np.percentile(rms_db, 5)),
            'signal_peak_db': float(np.percentile(rms_db, 95)),
            'dynamic_range': float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5)),
            'spectral_flatness': float(np.mean(spectral_flatness)),
            'spectral_centroid': float(np.mean(spectral_centroid)),
            'sample_rate': sample_rate,
            'duration': float(len(audio_data) / sample_rate),
            'rms_mean': float(np.mean(rms)),
            'rms_std': float(np.std(rms)),
        }
        
        del audio_data, rms, rms_db, spectral_flatness, spectral_centroid
        import gc
        gc.collect()
        
        return result
    except Exception as e:
        print(f"[SelfLearning] 特征分析失败: {e}")
        return {}


def learn_optimal_parameters(features: Dict, scene: str) -> Dict:
    history = load_history()

    similar = find_similar_records(features, history, top_n=3)

    if not similar:
        print("[SelfLearning] 无历史记录，使用默认参数")
        return {}

    print(f"[SelfLearning] 找到{len(similar)}个相似文件，正在学习最优参数...")

    # ============================================================
    # 评分逻辑：奖励"保留有效音频 + 适度移除明显静音"
    # ============================================================
    # 原逻辑缺陷：`if silence_removed > 0.1: score += silence_removed * 0.3`
    # 该公式奖励移除更多静音，导致系统偏好激进阈值（学到 -25.4dB 异常值）
    #
    # 新评分原则：
    #   1. 主权重保留率（kept_ratio）：保留 70%以上才高分，越高越好
    #   2. 静音移除率仅在 5%-25% 区间加分（避免过少=没效果，过多=过度移除）
    #   3. 移除率 >40% 直接判负分（防止过激参数被学习）
    # ============================================================
    best_params = {}
    best_score = float('-inf')

    for record in similar:
        params = record.parameters
        result = record.result

        kept_ratio = result.get('kept_ratio', 0)
        silence_removed = result.get('silence_removed_ratio', 0)

        score = 0.0
        # 主权重：保留率（越高越好，但需 >0.7 才认为合理）
        if kept_ratio >= 0.7:
            score += kept_ratio * 0.6
        elif kept_ratio >= 0.5:
            score += kept_ratio * 0.3
        else:
            # 保留率 <50% 视为过激移除，扣分
            score -= 0.5

        # 辅助权重：静音移除率在合理区间才加分
        if 0.05 <= silence_removed <= 0.25:
            score += 0.2
        elif silence_removed > 0.40:
            # 过度移除（>40%）扣分
            score -= 0.3

        if score > best_score:
            best_score = score
            best_params = params.copy()

    if not best_params:
        return {}

    # ============================================================
    # 参数合理性校验：过滤异常学到的 silence_threshold
    # ============================================================
    # 之前学到 -25.4dB 是因为评分错误。即便评分修正，仍需校验：
    #   silence_threshold 必须接近"人耳不可辨识阈值"
    #   即 noise_floor + [1dB, 12dB] 范围内（噪声掩盖到稍微宽松）
    #
    # 超出此范围视为异常值，丢弃 silence_threshold 字段
    # 但保留其他字段（noise_reduction、highpass_cutoff 等）
    # ============================================================
    noise_floor_db = features.get('noise_floor_db', -60.0)
    learned_threshold = best_params.get('silence_threshold')

    if learned_threshold is not None:
        min_reasonable = noise_floor_db + 1.0   # 至少高于噪声底噪 1dB
        max_reasonable = noise_floor_db + 12.0  # 不超过噪声底噪 + 12dB（人耳不可辨识范围上限）

        if not (min_reasonable <= learned_threshold <= max_reasonable):
            print(f"[SelfLearning] 学到的 silence_threshold={learned_threshold:.1f}dB 超出合理范围 "
                  f"[{min_reasonable:.1f}dB, {max_reasonable:.1f}dB]（noise_floor={noise_floor_db:.1f}dB），"
                  f"丢弃该字段，使用自适应计算值")
            best_params.pop('silence_threshold', None)
        else:
            print(f"[SelfLearning] 学到的 silence_threshold={learned_threshold:.1f}dB 在合理范围内，采用")

    # min_silence_duration 不再从历史学习——固定 1.5s 与基准线匹配
    best_params.pop('min_silence_duration', None)

    print(f"[SelfLearning] 学习到最优参数: {best_params}")
    return best_params


def record_processing(file_path: str, features: Dict, parameters: Dict, result: Dict):
    try:
        file_hash = compute_file_hash(file_path)
        history = load_history()
        
        existing = next((r for r in history if r.file_hash == file_hash), None)
        
        record = ProcessingRecord(
            file_hash=file_hash,
            features=features,
            parameters=parameters,
            result=result
        )
        
        if existing:
            idx = history.index(existing)
            history[idx] = record
        else:
            history.append(record)
        
        save_history(history)
        print(f"[SelfLearning] 记录已保存")
    except Exception as e:
        print(f"[SelfLearning] 保存记录失败: {e}")


def get_statistics() -> Dict:
    history = load_history()
    
    if not history:
        return {'total_records': 0}
    
    noise_levels = {}
    scenes = {}
    
    for r in history:
        nl = r.features.get('noise_level', 'unknown')
        noise_levels[nl] = noise_levels.get(nl, 0) + 1
        
        scene = r.parameters.get('scene', 'unknown')
        scenes[scene] = scenes.get(scene, 0) + 1
    
    return {
        'total_records': len(history),
        'noise_level_distribution': noise_levels,
        'scene_distribution': scenes,
    }