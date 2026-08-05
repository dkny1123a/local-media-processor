#!/usr/bin/env python3
"""
基于第一性原理的音频处理测试管道

第一性原理推导：
1. 录制场景：安静房间 → 噪声底 -52dB，信噪比高
2. 原始动态范围仅 5.8dB（录音设备AGC已压缩）→ 无需再压缩
3. 收听场景：骑行嘈杂 ~75dB SPL → 需要高响度（-14~-12 LUFS）
4. 核心问题：响度不足（原始 -40 LUFS），需 +26dB 增益
5. 高增益风险：噪声底被放大（-52+26=-26dB），但骑行环境可掩蔽

处理链设计：
- 高通滤波：去除 80Hz 以下低频瓮声
- 降噪：afftdn 去除稳态噪声（防止高增益放大噪声）
- 轻度压缩：仅在峰值过高时介入（保护人声）
- 高增益：+20~30dB（达到骑行收听响度）
- 软限制器：-1dB 安全网（防止 inter-sample peak 削波）

评分函数（基于收听场景）：
- 响度目标：-14 LUFS（骑行环境，非流媒体-16）
- 无削波（>-0.5dB）
- 无限制器持续介入（>-2dB < 10%）
- 增益平滑（>3dB 跳变 < 1%）
- 噪声门 < -30dB（骑行环境可掩蔽）
"""

import os
import sys
import re
import json
import subprocess
import shutil
import statistics
import numpy as np

# ============================================================
# 配置：基于第一性原理
# ============================================================
INPUT_DIR = "/Volumes/WangMovies/待处理"
WORK_DIR = "/tmp/audio_first_principles"
SAMPLE_DURATION = 30

# 收听场景约束（第一性原理推导）
TARGET_LOUDNESS = -14.0  # 骑行环境需要更高响度（非流媒体-16）
MAX_PEAK = -0.5  # 硬上限（防止 inter-sample peak 削波）
MAX_LIMITER_ENGAGEMENT = 0.10  # 限制器介入率上限（>10% 算失真）
MAX_GAIN_JUMP = 0.01  # 增益跳变率上限（>1% 算 pumping）
NOISE_FLOOR_TARGET = -30.0  # 噪声门目标（骑行环境可掩蔽）

# 评分权重
WEIGHTS = {
    'loudness_error': 0.30,      # 响度偏差（最重要：骑行必须听清）
    'clipping_rate': 0.25,       # 削波（硬指标：无削波）
    'limiter_engagement': 0.15,  # 限制器介入率（爆音根源）
    'gain_jump_rate': 0.15,      # 增益跳变率（pumping 根源）
    'noise_floor': 0.10,         # 噪声门（骑行掩蔽）
    'dynamic_range': 0.05,       # 动态范围（收听环境宽松）
}

# ============================================================
# 候选方案：基于第一性原理推导
# ============================================================
CANDIDATE_PLANS = {
    # 先压缩峰值，再提升增益（降低限制器介入）
    'P_compfirst_20': {
        'desc': '先compand压缩峰值 + volume+20dB',
        'af': 'highpass=f=80:p=2,afftdn=nr=15,'
              'compand=0.01:0.3:-80/-80|-30/-30|-20/-20|-10/-12|0/-8:1.5:0:-80:0.1,'
              'volume=20dB,aresample=22050,'
              'alimiter=limit=0.707:level=disabled:attack=10:release=100',
    },
    'P_compfirst_24': {
        'desc': '先compand强压峰值 + volume+24dB',
        'af': 'highpass=f=80:p=2,afftdn=nr=15,'
              'compand=0.01:0.3:-80/-80|-30/-30|-20/-22|-10/-15|0/-10:2:0:-80:0.1,'
              'volume=24dB,aresample=22050,'
              'alimiter=limit=0.707:level=disabled:attack=10:release=100',
    },
    # 适中增益 + 轻度压缩
    'P_mild_18': {
        'desc': '轻度compand + volume+18dB（适中）',
        'af': 'highpass=f=80:p=2,afftdn=nr=15,'
              'compand=0.01:0.1:-80/-80|-40/-40|-20/-18|-10/-10|0/-6:1.5:0:-80:0.1,'
              'volume=18dB,aresample=22050,'
              'alimiter=limit=0.707:level=disabled:attack=10:release=100',
    },
    # 两级compand + 适中增益（之前最优方案的改进版）
    'P_two_stage_18': {
        'desc': '两级compand + volume+18dB（降增益防限制器）',
        'af': 'highpass=f=80:p=2,afftdn=nr=15,'
              'compand=0.1:0.1:-80/-80|-60/-50|-40/-25|-20/-10|0/-3:3:0:-80:0.2,'
              'compand=0.01:0.01:-80/-80|-30/-10|0/-3:2:0:-80:0.1,'
              'volume=18dB,aresample=22050,'
              'alimiter=limit=0.707:level=disabled:attack=10:release=100',
    },
    # 两级compand + 高增益 + 低限制器阈值
    'P_two_stage_22_lowlimit': {
        'desc': '两级compand + volume+22dB + 限制器-3dB',
        'af': 'highpass=f=80:p=2,afftdn=nr=15,'
              'compand=0.1:0.1:-80/-80|-60/-50|-40/-25|-20/-10|0/-3:3:0:-80:0.2,'
              'compand=0.01:0.01:-80/-80|-30/-10|0/-3:2:0:-80:0.1,'
              'volume=22dB,aresample=22050,'
              'alimiter=limit=0.707:level=disabled:attack=15:release=150',
    },
    # 无压缩 + 适中增益（信任原始DR）
    'P_pure_gain_20': {
        'desc': '纯增益+20dB（无压缩，信任原始DR=5.8dB）',
        'af': 'highpass=f=80:p=2,afftdn=nr=15,'
              'volume=20dB,aresample=22050,'
              'alimiter=limit=0.707:level=disabled:attack=10:release=100',
    },
}


def find_samples(input_dir, quick=False):
    exts = ('.m4a', '.mp3', '.wav', '.aac', '.flac', '.ogg')
    samples = []
    for f in sorted(os.listdir(input_dir)):
        if f.startswith('._'):
            continue
        if f.lower().endswith(exts):
            full = os.path.join(input_dir, f)
            if os.path.getsize(full) > 100 * 1024:
                samples.append(full)
    if quick and len(samples) > 5:
        step = len(samples) // 5
        samples = [samples[i * step] for i in range(5)]
    return samples


def extract_sample_clip(input_path, duration, work_dir):
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', input_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return None
    try:
        total = float(json.loads(result.stdout)['format']['duration'])
    except (json.JSONDecodeError, KeyError, ValueError):
        return None

    start = max(0, (total - duration) / 2)
    if total < duration:
        start = 0
        duration = total

    clip_path = os.path.join(work_dir, 'clips', os.path.basename(input_path) + '.wav')
    os.makedirs(os.path.dirname(clip_path), exist_ok=True)

    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error',
        '-ss', str(start),
        '-t', str(duration),
        '-i', input_path,
        '-ac', '1',
        '-ar', '22050',
        '-c:a', 'pcm_s16le',
        '-y', clip_path
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    if result.returncode != 0:
        return None
    return clip_path


def process_with_plan(clip_path, plan_af, output_path):
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error',
        '-i', clip_path,
        '-af', plan_af,
        '-ac', '1',
        '-ar', '22050',
        '-c:a', 'libmp3lame',
        '-b:a', '96k',
        '-y', output_path
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    return result.returncode == 0


def analyze_audio(file_path):
    """分析 6 维指标"""
    metrics = {}

    # loudnorm 分析
    cmd = ['ffmpeg', '-hide_banner', '-i', file_path,
           '-af', 'loudnorm=print_format=json', '-f', 'null', '-']
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    json_match = re.search(r'\{[^}]+\}', result.stderr, re.DOTALL)
    if json_match:
        try:
            ln = json.loads(json_match.group(0))
            metrics['input_i'] = float(ln.get('input_i', -23.0))
            metrics['input_tp'] = float(ln.get('input_tp', -2.0))
            metrics['input_lra'] = float(ln.get('input_lra', 11.0))
            metrics['input_thresh'] = float(ln.get('input_thresh', -33.0))
        except (json.JSONDecodeError, ValueError):
            metrics.update({'input_i': -23.0, 'input_tp': -2.0,
                            'input_lra': 11.0, 'input_thresh': -33.0})
    else:
        metrics.update({'input_i': -23.0, 'input_tp': -2.0,
                        'input_lra': 11.0, 'input_thresh': -33.0})

    # 峰值分布 + RMS 跳变
    cmd = ['ffmpeg', '-hide_banner', '-i', file_path,
           '-af', 'astats=metadata=1:reset=0.5,ametadata=mode=print:file=-',
           '-f', 'null', '-']
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    peaks = []
    rms_vals = []
    current_pts = None
    for line in result.stdout.split('\n'):
        pts_m = re.search(r'pts_time:([0-9.]+)', line)
        if pts_m:
            current_pts = float(pts_m.group(1))
        peak_m = re.search(r'Peak_level=(-?[0-9.inf-]+)', line)
        if peak_m and current_pts is not None:
            v = peak_m.group(1)
            if v != '-inf':
                peaks.append(float(v))
        rms_m = re.search(r'RMS_level=(-?[0-9.inf-]+)', line)
        if rms_m and current_pts is not None:
            v = rms_m.group(1)
            if v != '-inf':
                rms_vals.append(float(v))

    if peaks:
        metrics['clipping_rate'] = len([p for p in peaks if p > -0.5]) / len(peaks)
        metrics['limiter_engagement'] = len([p for p in peaks if p > -2.0]) / len(peaks)
        metrics['max_peak'] = max(peaks)
    else:
        metrics['clipping_rate'] = 1.0
        metrics['limiter_engagement'] = 1.0
        metrics['max_peak'] = 0.0

    if len(rms_vals) > 1:
        jumps = [abs(rms_vals[i + 1] - rms_vals[i]) for i in range(len(rms_vals) - 1)]
        metrics['gain_jump_rate'] = len([j for j in jumps if j > 3.0]) / len(jumps)
        metrics['max_gain_jump'] = max(jumps)
    else:
        metrics['gain_jump_rate'] = 1.0
        metrics['max_gain_jump'] = 99.0

    return metrics


def score_metrics(metrics):
    """基于第一性原理的评分函数"""
    scores = {}

    # 响度偏差（目标-14 LUFS，骑行环境）
    loudness_err = abs(metrics['input_i'] - TARGET_LOUDNESS)
    scores['loudness_error'] = max(0, 100 - loudness_err * 8)

    # 削波率（硬指标：0=完美，>0=严重）
    scores['clipping_rate'] = 100 * (1 - min(1.0, metrics['clipping_rate'] * 20))

    # 限制器介入率（爆音根源：0=完美，>10%=0分）
    scores['limiter_engagement'] = 100 * max(0, 1 - metrics['limiter_engagement'] / MAX_LIMITER_ENGAGEMENT)

    # 增益跳变率（pumping根源：0=完美，>1%=0分）
    scores['gain_jump_rate'] = 100 * max(0, 1 - metrics['gain_jump_rate'] / MAX_GAIN_JUMP)

    # 噪声门（骑行环境可掩蔽，< -30=满分）
    thresh = metrics['input_thresh']
    if thresh < NOISE_FLOOR_TARGET:
        scores['noise_floor'] = 100
    elif thresh > -15:
        scores['noise_floor'] = 0
    else:
        scores['noise_floor'] = (-15 - thresh) / (-15 - NOISE_FLOOR_TARGET) * 100

    # 动态范围（骑行环境宽松，5-25均可）
    lra = metrics['input_lra']
    if 5 <= lra <= 25:
        scores['dynamic_range'] = 100
    elif 3 <= lra < 5:
        scores['dynamic_range'] = (lra - 3) / 2 * 100
    elif 25 < lra <= 35:
        scores['dynamic_range'] = (35 - lra) / 10 * 100
    else:
        scores['dynamic_range'] = 0

    total = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)
    return scores, total


def run_test_plan(clip_path, plan_name, plan):
    output_path = os.path.join(os.path.dirname(clip_path),
                               f"{plan_name}_{os.path.basename(clip_path)}")
    success = process_with_plan(clip_path, plan['af'], output_path)
    if not success:
        return None

    metrics = analyze_audio(output_path)
    scores, total = score_metrics(metrics)

    if os.path.exists(output_path):
        os.unlink(output_path)

    return {
        'plan': plan_name,
        'desc': plan['desc'],
        'metrics': metrics,
        'scores': scores,
        'total_score': total,
    }


def main():
    quick = '--quick' in sys.argv

    print("=" * 70)
    print("基于第一性原理的音频处理测试管道")
    print("=" * 70)
    print(f"录制场景: 安静房间（噪声底 -52dB）")
    print(f"收听场景: 骑行嘈杂（~75dB SPL）")
    print(f"目标响度: {TARGET_LOUDNESS} LUFS（骑行环境，非流媒体-16）")
    print(f"核心问题: 响度不足（原始 -40 LUFS，需 +26dB）")
    print(f"候选方案: {len(CANDIDATE_PLANS)} 个")
    print(f"评分维度: 6 维（权重: {WEIGHTS}）")
    print("=" * 70)

    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    os.makedirs(WORK_DIR, exist_ok=True)

    # 1. 收集样本
    print("\n[1/4] 收集样本...")
    samples = find_samples(INPUT_DIR, quick=quick)
    print(f"  样本数: {len(samples)}")

    clips = []
    for s in samples:
        clip = extract_sample_clip(s, SAMPLE_DURATION, WORK_DIR)
        if clip:
            clips.append((os.path.basename(s), clip))
            print(f"  ✓ {os.path.basename(s)}")

    # 2. 测试所有方案
    print(f"\n[2/4] 测试 {len(CANDIDATE_PLANS)} 个方案 × {len(clips)} 个样本...")
    all_results = {}

    for plan_name, plan in CANDIDATE_PLANS.items():
        print(f"\n  方案 {plan_name}: {plan['desc']}")
        plan_results = []

        for clip_name, clip_path in clips:
            result = run_test_plan(clip_path, plan_name, plan)
            if result:
                plan_results.append((clip_name, result))
                print(f"    ✓ {clip_name}: 评分 {result['total_score']:.1f} "
                      f"(i={result['metrics']['input_i']:.1f}, tp={result['metrics']['input_tp']:.2f})")
            else:
                print(f"    ✗ {clip_name}: 处理失败")

        all_results[plan_name] = plan_results

    # 3. 汇总
    print(f"\n[3/4] 汇总评分...")
    summary = {}
    for plan_name, results in all_results.items():
        if not results:
            continue
        avg_score = sum(r['total_score'] for _, r in results) / len(results)
        min_score = min(r['total_score'] for _, r in results)
        max_score = max(r['total_score'] for _, r in results)

        avg_metrics = {}
        for key in all_results[plan_name][0][1]['metrics']:
            vals = [r['metrics'][key] for _, r in results if key in r['metrics']]
            if vals:
                avg_metrics[key] = sum(vals) / len(vals)

        summary[plan_name] = {
            'desc': CANDIDATE_PLANS[plan_name]['desc'],
            'avg_score': avg_score,
            'min_score': min_score,
            'max_score': max_score,
            'avg_metrics': avg_metrics,
            'sample_count': len(results),
        }

    # 4. 排名
    print(f"\n[4/4] 综合排名...")
    print("\n" + "=" * 90)
    print(f"{'排名':<4} {'方案':<28} {'描述':<38} {'平均分':<8} {'最低分':<8} {'最高分':<8}")
    print("-" * 90)

    ranked = sorted(summary.items(), key=lambda x: -x[1]['avg_score'])
    for i, (plan_name, info) in enumerate(ranked, 1):
        print(f"{i:<4} {plan_name:<28} {info['desc']:<38} {info['avg_score']:<8.1f} "
              f"{info['min_score']:<8.1f} {info['max_score']:<8.1f}")

    # 最优方案详情
    best_plan, best_info = ranked[0]
    print(f"\n{'=' * 90}")
    print(f"最优方案: {best_plan} - {best_info['desc']}")
    print(f"{'=' * 90}")
    print(f"综合评分: {best_info['avg_score']:.1f} (范围 {best_info['min_score']:.1f}-{best_info['max_score']:.1f})")
    print(f"\n平均指标:")
    for k, v in best_info['avg_metrics'].items():
        print(f"  {k:<25}: {v:.2f}")

    # 6 维评分明细
    print(f"\n{'=' * 90}")
    print(f"最优方案 6 维评分明细: {best_plan}")
    print(f"{'=' * 90}")
    print(f"{'维度':<25} {'权重':<8} {'平均分':<10} {'说明'}")
    print("-" * 90)
    for dim in WEIGHTS:
        dim_scores = []
        for clip_name, result in all_results[best_plan]:
            if dim in result['scores']:
                dim_scores.append(result['scores'][dim])
        if dim_scores:
            avg = sum(dim_scores) / len(dim_scores)
            desc = {
                'loudness_error': f'目标{TARGET_LOUDNESS}LUFS',
                'clipping_rate': '硬指标（>0=严重）',
                'limiter_engagement': f'上限{MAX_LIMITER_ENGAGEMENT*100}%',
                'gain_jump_rate': f'上限{MAX_GAIN_JUMP*100}%',
                'noise_floor': f'目标<{NOISE_FLOOR_TARGET}dB',
                'dynamic_range': '骑行环境5-25均可',
            }.get(dim, '')
            print(f"{dim:<25} {WEIGHTS[dim]:<8.2f} {avg:<10.1f} {desc}")

    # 样本对比
    print(f"\n{'=' * 90}")
    print("各样本评分对比")
    print(f"{'=' * 90}")
    print(f"{'样本':<28}", end='')
    for plan_name in CANDIDATE_PLANS:
        print(f" {plan_name[:14]:<15}", end='')
    print()
    print("-" * 90)

    for clip_name, _ in clips:
        print(f"{clip_name[:28]:<28}", end='')
        for plan_name in CANDIDATE_PLANS:
            results = all_results.get(plan_name, [])
            score = next((r['total_score'] for cn, r in results if cn == clip_name), None)
            if score is not None:
                print(f" {score:<15.1f}", end='')
            else:
                print(f" {'FAIL':<15}", end='')
        print()

    # 保存报告
    report_path = os.path.join(WORK_DIR, 'first_principles_report.json')
    report = {
        'target_loudness': TARGET_LOUDNESS,
        'weights': WEIGHTS,
        'sample_count': len(clips),
        'plans': {},
    }
    for plan_name, results in all_results.items():
        report['plans'][plan_name] = {
            'desc': CANDIDATE_PLANS[plan_name]['desc'],
            'avg_score': summary.get(plan_name, {}).get('avg_score', 0),
            'results': [{'sample': cn, **r} for cn, r in results],
        }
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n详细报告: {report_path}")

    shutil.rmtree(os.path.join(WORK_DIR, 'clips'), ignore_errors=True)


if __name__ == '__main__':
    main()
