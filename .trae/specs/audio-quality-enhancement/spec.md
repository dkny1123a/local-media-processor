# 骑行音频处理质量提升 - 产品需求文档

## Overview
- **Summary**: 针对城市道路骑行场景下人声被风噪和交通噪音淹没的问题，对音频处理流程进行深度优化，提升语音可懂度和降噪效果。
- **Purpose**: 解决实际路测中发现的问题：城市道路上人声较小盖不过风噪和道路噪音；公园绿道人声基本能听清楚。通过优化降噪算法、修复智能度增强逻辑、扩展滤波范围等方式提升城市场景下的语音质量。
- **Target Users**: 骑行时使用蓝牙耳机收听播客/有声书的用户

## Goals
- 修复 `apply_intelligibility_boost` 逻辑错误，不再放大噪声帧
- 优化降噪处理，适配交通非稳态噪声
- 扩展带通滤波范围，保留语音高频辅音信息
- 增强语音频段增益，提升语音可懂度
- 添加基于VAD的自适应增益（语音门控）

## Non-Goals (Out of Scope)
- 不改变现有的分块处理架构
- 不引入新的重量级深度学习模型（保持Silero VAD为主）
- 不修改前端播放器逻辑
- 不改变文件输出格式（保持16kHz单声道MP3）

## Background & Context
- 现有代码使用 `noisereduce` 默认参数（稳态噪声假设），但交通噪声是非稳态的
- 带通滤波器截止频率 300-3400Hz 过窄，丢失了语音高频辅音信息
- `apply_intelligibility_boost` 逻辑有误：对低dB帧放大实际是放大噪声
- 高通滤波仅100Hz，不足以有效抑制风噪（风噪主要集中在20-500Hz）
- 城市道路场景需要更激进的语音提升策略

## Functional Requirements
- **FR-1**: 修复智能度增强逻辑，改为对语音频段内的中低音量信号做提升，对纯噪声帧做衰减
- **FR-2**: 降噪处理支持非稳态噪声模式（stationary=False），并调整处理顺序为先降噪再滤波
- **FR-3**: 扩展带通滤波范围到80-8000Hz，高通滤波截止频率提高到150Hz
- **FR-4**: 增强语音频段增益（从0.5提高到1.0-1.5），添加高频辅音区域提升
- **FR-5**: 添加基于VAD的自适应增益，对语音段做额外增益，对非语音段做衰减

## Non-Functional Requirements
- **NFR-1**: 处理时间增加不超过30%
- **NFR-2**: 内存使用不超过现有水平的120%
- **NFR-3**: 保持与现有API兼容，无需修改前端

## Constraints
- **Technical**: Python 3.10, noisereduce 2.0.1, librosa, scipy
- **Dependencies**: 保持现有依赖，不引入新的重量级库

## Assumptions
- 用户使用蓝牙耳机收听，采样率16kHz足够
- Silero VAD模型已正确加载并可用
- 城市场景下噪声水平高于公园绿道场景

## Acceptance Criteria

### AC-1: 智能度增强逻辑修复
- **Given**: 音频中包含低音量噪声帧（dB<-35）和中低音量语音帧（dB在-25到-10之间）
- **When**: 应用智能度增强处理
- **Then**: 语音帧获得6-10dB增益，噪声帧被衰减而非放大
- **Verification**: `programmatic`

### AC-2: 非稳态降噪
- **Given**: 音频中包含交通噪声（非稳态）
- **When**: 应用降噪处理
- **Then**: 降噪效果显著提升，语音清晰度改善
- **Verification**: `human-judgment`

### AC-3: 扩展带通滤波范围
- **Given**: 音频采样率为16kHz或更高
- **When**: 应用带通滤波
- **Then**: 保留80-8000Hz频段信号，有效抑制低频风噪和高频噪声
- **Verification**: `programmatic`

### AC-4: 语音频段增益增强
- **Given**: 音频中包含语音信号
- **When**: 应用语音增强处理
- **Then**: 300-3400Hz语音频段获得1.0-1.5倍增益，4000-8000Hz获得0.3-0.5倍增益
- **Verification**: `programmatic`

### AC-5: VAD自适应增益（语音门控）
- **Given**: Silero VAD已正确识别语音段和非语音段
- **When**: 应用VAD自适应增益处理
- **Then**: 语音段获得额外6-10dB增益，非语音段获得-6dB衰减
- **Verification**: `programmatic`

## Open Questions
- [ ] 是否需要根据场景（城市/公园）动态调整参数？
- [ ] 语音门控的增益值是否需要用户可配置？
