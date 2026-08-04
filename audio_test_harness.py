#!/usr/bin/env python3
"""
音频处理科学测试管道

方法学：
1. 黄金样本集：从每个原始样本截取 30s 代表片段（覆盖多种录音场景）
2. 6 维客观指标：响度偏差、削波率、限制器介入率、动态范围、增益跳变、噪声门
3. 综合评分函数：多目标加权评分，避免单点优化
4. 参数网格搜索：在候选参数空间中找出全局最优
5. A/B 对比报告：客观指标对比，而非主观试听

用法：
    python3 audio_test_harness.py              # 运行完整测试
    python3 audio_test_harness.py --quick       # 仅 5 个样本快速测试
"""

import os
import sys
import re
import json
import subprocess
import tempfile
import shutil
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# 配置
# ============================================================
INPUT_DIR = "/Volumes/WangMovies/待处理"
WORK_DIR = "/tmp/audio_test_harness"
SAMPLE_DURATION = 30  # 每个样本截取 30s
TARGET_LOUDNESS = -16.0  # 目标响度 LUFS

# 6 维评分权重（总和=1.0）
WEIGHTS = {
    'loudness_error': 0.20,   # 响度偏差 |input_i - (-16)|
    'clipping_rate': 0.25,   # 削波率（>-0.5dB 占比）
    'limiter_engagement': 0.20,  # 限制器介入率（>-2dB 占比）
    'dynamic_range': 0.10,   # 动态范围（LRA 10-20 最佳）
    'gain_jump_rate': 0.15,  # 增益跳变率（>3dB 跳变占比）
    'noise_floor': 0.10,     # 噪声门（thresh 值）
}

# ============================================================
# 候选方案（参数网格）
# ============================================================
CANDIDATE_PLANS = {
    'A_compand_vol6': {
        'desc': 'compand温和压缩 + volume+6dB',
        'af': 'highpass=f=80:p=2,afftdn=nr=12,'
              'compand=0.1:0.1:-80/-80|-50/-50|-30/-15|-10/-5|0/-3:3:0:-80:0.2,'
              'volume=6dB,aresample=22050,'
              'alimiter=limit=0.707:level=disabled:attack=10:release=100',
    },
    'G_compand_vol12': {
        'desc': 'compand + volume+12dB（高增益）',
        'af': 'highpass=f=80:p=2,afftdn=nr=12,'
              'compand=0.1:0.1:-80/-80|-50/-50|-30/-15|-10/-5|0/-3:3:0:-80:0.2,'
              'volume=12dB,aresample=22050,'
              'alimiter=limit=0.707:level=disabled:attack=10:release=100',
    },
    'H_compand_vol15': {
        'desc': 'compand + volume+15dB（最高增益）',
        'af': 'highpass=f=80:p=2,afftdn=nr=12,'
              'compand=0.1:0.1:-80/-80|-50/-50|-30/-15|-10/-5|0/-3:3:0:-80:0.2,'
              'volume=15dB,aresample=22050,'
              'alimiter=limit=0.707:level=disabled:attack=10:release=100',
    },
    'I_vol_first_compand': {
        'desc': 'volume+10dB前置 + compand后置',
        'af': 'highpass=f=80:p=2,afftdn=nr=12,'
              'volume=10dB,'
              'compand=0.1:0.1:-80/-80|-50/-50|-30/-15|-10/-5|0/-3:3:0:-80:0.2,'
              'aresample=22050,'
              'alimiter=limit=0.707:level=disabled:attack=10:release=100',
    },
    'J_aggressive_compand': {
        'desc': '激进compand提升低电平信号',
        'af': 'highpass=f=80:p=2,afftdn=nr=12,'
              'compand=0.1:0.1:-80/-80|-60/-50|-50/-35|-40/-25|-30/-15|-20/-8|-10/-3|0/-1:4:0:-80:0.2,'
              'volume=6dB,aresample=22050,'
              'alimiter=limit=0.707:level=disabled:attack=10:release=100',
    },
    'K_compand_chain': {
        'desc': '两级compand + volume+10dB',
        'af': 'highpass=f=80:p=2,afftdn=nr=12,'
              'compand=0.1:0.1:-80/-80|-60/-50|-40/-25|-20/-10|0/-3:3:0:-80:0.2,'
              'compand=0.01:0.01:-80/-80|-30/-10|0/-3:2:0:-80:0.1,'
              'volume=10dB,aresample=22050,'
              'alimiter=limit=0.707:level=disabled:attack=10:release=100',
    },
}


def find_samples(input_dir, quick=False):
    """查找所有样本文件"""
    exts = ('.m4a', '.mp3', '.wav', '.aac', '.flac', '.ogg')
    samples = []
    for f in sorted(os.listdir(input_dir)):
        if f.startswith('._'):
            continue
        if f.lower().endswith(exts):
            full = os.path.join(input_dir, f)
            if os.path.getsize(full) > 100 * 1024:
                samples.append(full)
    if quick:
        # 快速模式：取 5 个代表性样本
        if len(samples) > 5:
            step = len(samples) // 5
            samples = [samples[i * step] for i in range(5)]
    return samples


def extract_sample_clip(input_path, duration, work_dir):
    """从输入文件截取 30s 代表片段（取中间位置）"""
    # 获取总时长
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', input_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return None
    try:
        info = json.loads(result.stdout)
        total = float(info['format']['duration'])
    except (json.JSONDecodeError, KeyError, ValueError):
        return None

    # 取中间 30s
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
    """用指定方案处理音频"""
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
    """分析音频文件的 6 维指标"""
    metrics = {}

    # 1. loudnorm 分析（响度、峰值、动态范围、噪声门）
    cmd = ['ffmpeg', '-hide_banner', '-i', file_path,
           '-af', 'loudnorm=print_format=json', '-f', 'null', '-']
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    json_match = re.search(r'\{[^}]+\}', result.stderr, re.DOTALL)
    if json_match:
        try:
            ln = json.loads(json_match.group(0))
            input_i = float(ln.get('input_i', -23.0))
            input_tp = float(ln.get('input_tp', -2.0))
            input_lra = float(ln.get('input_lra', 11.0))
            input_thresh = float(ln.get('input_thresh', -33.0))
        except (json.JSONDecodeError, ValueError):
            input_i, input_tp, input_lra, input_thresh = -23.0, -2.0, 11.0, -33.0
    else:
        input_i, input_tp, input_lra, input_thresh = -23.0, -2.0, 11.0, -33.0

    metrics['input_i'] = input_i
    metrics['input_tp'] = input_tp
    metrics['input_lra'] = input_lra
    metrics['input_thresh'] = input_thresh

    # 2-3. 峰值分布分析（削波率、限制器介入率）
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

    # 4. 增益跳变率
    if len(rms_vals) > 1:
        jumps = [abs(rms_vals[i + 1] - rms_vals[i]) for i in range(len(rms_vals) - 1)]
        metrics['gain_jump_rate'] = len([j for j in jumps if j > 3.0]) / len(jumps)
        metrics['max_gain_jump'] = max(jumps)
    else:
        metrics['gain_jump_rate'] = 1.0
        metrics['max_gain_jump'] = 99.0

    return metrics


def score_metrics(metrics):
    """计算 6 维综合评分（0-100，越高越好）"""
    scores = {}

    # 1. 响度偏差（|input_i - (-16)|，0=完美，>10=0分）
    loudness_err = abs(metrics['input_i'] - TARGET_LOUDNESS)
    scores['loudness_error'] = max(0, 100 - loudness_err * 10)

    # 2. 削波率（0=完美，>0=0分）
    scores['clipping_rate'] = 100 * (1 - min(1.0, metrics['clipping_rate'] * 10))

    # 3. 限制器介入率（0=完美，>50%=0分）
    scores['limiter_engagement'] = 100 * (1 - min(1.0, metrics['limiter_engagement'] * 2))

    # 4. 动态范围（LRA 10-20=100分，<5或>30=0分）
    lra = metrics['input_lra']
    if 10 <= lra <= 20:
        scores['dynamic_range'] = 100
    elif 5 <= lra < 10:
        scores['dynamic_range'] = (lra - 5) / 5 * 100
    elif 20 < lra <= 30:
        scores['dynamic_range'] = (30 - lra) / 10 * 100
    else:
        scores['dynamic_range'] = 0

    # 5. 增益跳变率（0=完美，>5%=0分）
    scores['gain_jump_rate'] = 100 * (1 - min(1.0, metrics['gain_jump_rate'] * 20))

    # 6. 噪声门（thresh < -40=100分，> -20=0分）
    thresh = metrics['input_thresh']
    if thresh < -40:
        scores['noise_floor'] = 100
    elif thresh > -20:
        scores['noise_floor'] = 0
    else:
        scores['noise_floor'] = (-20 - thresh) / 20 * 100

    # 加权综合评分
    total = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)
    return scores, total


def run_test_plan(clip_path, plan_name, plan):
    """在单个样本上测试一个方案"""
    output_path = os.path.join(os.path.dirname(clip_path),
                               f"{plan_name}_{os.path.basename(clip_path)}")
    success = process_with_plan(clip_path, plan['af'], output_path)
    if not success:
        return None

    metrics = analyze_audio(output_path)
    scores, total = score_metrics(metrics)

    # 清理输出文件
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
    print("音频处理科学测试管道")
    print("=" * 70)
    print(f"目标响度: {TARGET_LOUDNESS} LUFS")
    print(f"样本时长: {SAMPLE_DURATION}s 每个文件")
    print(f"候选方案: {len(CANDIDATE_PLANS)} 个")
    print(f"评分维度: 6 维（权重: {WEIGHTS}）")
    if quick:
        print("模式: 快速（5 个样本）")
    print("=" * 70)

    # 准备工作目录
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    os.makedirs(WORK_DIR, exist_ok=True)

    # 1. 收集样本并截取片段
    print("\n[1/4] 收集样本并截取代表片段...")
    samples = find_samples(INPUT_DIR, quick=quick)
    print(f"  样本数: {len(samples)}")

    clips = []
    for s in samples:
        clip = extract_sample_clip(s, SAMPLE_DURATION, WORK_DIR)
        if clip:
            clips.append((os.path.basename(s), clip))
            print(f"  ✓ {os.path.basename(s)}")
        else:
            print(f"  ✗ {os.path.basename(s)} (截取失败)")

    print(f"  有效片段: {len(clips)}")

    # 2. 在所有样本上测试所有方案
    print(f"\n[2/4] 测试 {len(CANDIDATE_PLANS)} 个方案 × {len(clips)} 个样本...")
    all_results = {}

    for plan_name, plan in CANDIDATE_PLANS.items():
        print(f"\n  方案 {plan_name}: {plan['desc']}")
        plan_results = []

        for clip_name, clip_path in clips:
            result = run_test_plan(clip_path, plan_name, plan)
            if result:
                plan_results.append((clip_name, result))
                print(f"    ✓ {clip_name}: 评分 {result['total_score']:.1f}")
            else:
                print(f"    ✗ {clip_name}: 处理失败")

        all_results[plan_name] = plan_results

    # 3. 汇总评分
    print(f"\n[3/4] 汇总评分...")
    summary = {}
    for plan_name, results in all_results.items():
        if not results:
            continue
        avg_score = sum(r['total_score'] for _, r in results) / len(results)
        min_score = min(r['total_score'] for _, r in results)
        max_score = max(r['total_score'] for _, r in results)

        # 平均指标
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

    # 4. 排名并输出报告
    print(f"\n[4/4] 生成报告...")
    print("\n" + "=" * 70)
    print("综合评分排名（平均分降序）")
    print("=" * 70)
    print(f"{'排名':<4} {'方案':<25} {'描述':<35} {'平均分':<8} {'最低分':<8} {'最高分':<8}")
    print("-" * 70)

    ranked = sorted(summary.items(), key=lambda x: -x[1]['avg_score'])
    for i, (plan_name, info) in enumerate(ranked, 1):
        print(f"{i:<4} {plan_name:<25} {info['desc']:<35} {info['avg_score']:<8.1f} "
              f"{info['min_score']:<8.1f} {info['max_score']:<8.1f}")

    # 最优方案详细指标
    best_plan, best_info = ranked[0]
    print(f"\n{'=' * 70}")
    print(f"最优方案: {best_plan} - {best_info['desc']}")
    print(f"{'=' * 70}")
    print(f"综合评分: {best_info['avg_score']:.1f} (范围 {best_info['min_score']:.1f}-{best_info['max_score']:.1f})")
    print(f"\n平均指标:")
    for k, v in best_info['avg_metrics'].items():
        print(f"  {k:<25}: {v:.2f}")

    # 按样本对比各方案
    print(f"\n{'=' * 70}")
    print("各样本上的方案对比")
    print(f"{'=' * 70}")
    print(f"{'样本':<30}", end='')
    for plan_name in CANDIDATE_PLANS:
        print(f" {plan_name[:12]:<13}", end='')
    print()
    print("-" * 70)

    for clip_name, _ in clips:
        print(f"{clip_name[:30]:<30}", end='')
        for plan_name in CANDIDATE_PLANS:
            results = all_results.get(plan_name, [])
            score = next((r['total_score'] for cn, r in results if cn == clip_name), None)
            if score is not None:
                print(f" {score:<13.1f}", end='')
            else:
                print(f" {'FAIL':<13}", end='')
        print()

    # 6 维评分明细（最优方案）
    print(f"\n{'=' * 70}")
    print(f"最优方案 6 维评分明细: {best_plan}")
    print(f"{'=' * 70}")
    print(f"{'维度':<25} {'权重':<8} {'平均分':<10}")
    print("-" * 70)
    for dim in WEIGHTS:
        dim_scores = []
        for clip_name, result in all_results[best_plan]:
            if dim in result['scores']:
                dim_scores.append(result['scores'][dim])
        if dim_scores:
            avg = sum(dim_scores) / len(dim_scores)
            print(f"{dim:<25} {WEIGHTS[dim]:<8.2f} {avg:<10.1f}")

    print(f"\n{'=' * 70}")
    print("测试完成")
    print(f"{'=' * 70}")

    # 保存 JSON 报告
    report_path = os.path.join(WORK_DIR, 'test_report.json')
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
    print(f"\n详细报告已保存: {report_path}")

    # 清理片段
    shutil.rmtree(os.path.join(WORK_DIR, 'clips'), ignore_errors=True)


if __name__ == '__main__':
    main()
