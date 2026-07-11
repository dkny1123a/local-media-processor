# 骑行音频处理质量提升 - 实现计划

## [x] Task 1: 修复智能度增强逻辑 (apply_intelligibility_boost)
## [x] Task 2: 优化降噪处理 - 非稳态噪声模式
## [x] Task 3: 扩展带通滤波范围
## [x] Task 4: 增强语音频段增益
## [x] Task 5: 添加VAD自适应增益（语音门控）
## [x] Task 6: 测试与验证
- **Priority**: high
- **Depends On**: None
- **Description**: 
  - 修改 `apply_intelligibility_boost` 函数，修复逻辑错误
  - 对语音频段内的中低音量信号（dB在-25到-10之间）做6-10dB增益
  - 对低音量噪声帧（dB<-35）做衰减而非放大
  - 使用STFT进行频段分析，仅在语音频段内应用增强
- **Acceptance Criteria Addressed**: AC-1
- **Test Requirements**:
  - `programmatic` TR-1.1: 输入包含低音量噪声帧(dB<-35)和中低音量语音帧(dB在-25到-10之间)的音频，验证噪声帧被衰减、语音帧被提升
  - `programmatic` TR-1.2: 验证处理后音频不出现削波(最大值不超过1.0)
- **Notes**: 需要使用STFT分析频段，确保只在语音频段内应用增强

## [ ] Task 2: 优化降噪处理 - 非稳态噪声模式
- **Priority**: high
- **Depends On**: None
- **Description**: 
  - 修改 `process_single_cycling_chunk` 中的降噪调用，添加 `stationary=False` 参数
  - 调整处理顺序：先降噪再带通滤波（当前是先滤波再降噪）
  - 确保自适应处理器中也使用非稳态模式
- **Acceptance Criteria Addressed**: AC-2
- **Test Requirements**:
  - `programmatic` TR-2.1: 验证降噪调用中包含 `stationary=False` 参数
  - `human-judgment` TR-2.2: 使用包含交通噪声的样本测试，验证降噪效果提升
- **Notes**: 非稳态降噪计算量更大，需要监控处理时间

## [ ] Task 3: 扩展带通滤波范围
- **Priority**: high
- **Depends On**: None
- **Description**: 
  - 修改 `apply_bandpass_filter` 默认参数：low_cut=80.0, high_cut=8000.0
  - 修改 `apply_voice_enhancement` 中的语音频段范围：扩展到80-8000Hz
  - 修改自适应处理器中高通滤波截止频率从100Hz提高到150Hz
- **Acceptance Criteria Addressed**: AC-3
- **Test Requirements**:
  - `programmatic` TR-3.1: 验证带通滤波器参数为80-8000Hz
  - `programmatic` TR-3.2: 验证高通滤波截止频率为150Hz
- **Notes**: 需要确保采样率足够（16kHz时8000Hz是奈奎斯特频率的一半）

## [ ] Task 4: 增强语音频段增益
- **Priority**: high
- **Depends On**: Task 3
- **Description**: 
  - 修改 `apply_voice_enhancement`：将语音频段(300-3400Hz)增益从0.5提高到1.0-1.5
  - 添加对4000-8000Hz高频辅音区域的适度提升（0.3-0.5）
  - 调整过渡频段的增益曲线，使频率响应更平滑
- **Acceptance Criteria Addressed**: AC-4
- **Test Requirements**:
  - `programmatic` TR-4.1: 验证300-3400Hz频段增益为1.0-1.5
  - `programmatic` TR-4.2: 验证4000-8000Hz频段增益为0.3-0.5
- **Notes**: 需要平衡增益强度，避免过度提升导致失真

## [ ] Task 5: 添加VAD自适应增益（语音门控）
- **Priority**: medium
- **Depends On**: Task 1-4
- **Description**: 
  - 在 `cycling_audio_processor.py` 中添加新函数 `apply_vad_gate`
  - 使用Silero VAD识别语音段和非语音段
  - 对语音段做额外6-10dB增益，对非语音段做-6dB衰减
  - 在 `process_single_cycling_chunk` 中集成该功能
- **Acceptance Criteria Addressed**: AC-5
- **Test Requirements**:
  - `programmatic` TR-5.1: 验证VAD识别的语音段获得额外增益
  - `programmatic` TR-5.2: 验证非语音段获得衰减
- **Notes**: 需要确保VAD模型已正确加载，处理时间增加不超过30%

## [ ] Task 6: 测试与验证
- **Priority**: high
- **Depends On**: Task 1-5
- **Description**: 
  - 使用城市场景样本进行测试
  - 验证处理时间增加不超过30%
  - 验证内存使用不超过现有水平的120%
  - 验证输出格式保持16kHz单声道MP3
- **Acceptance Criteria Addressed**: 所有AC
- **Test Requirements**:
  - `programmatic` TR-6.1: 测试处理时间
  - `programmatic` TR-6.2: 测试输出文件格式
  - `human-judgment` TR-6.3: 主观评估语音清晰度
- **Notes**: 需要使用实际骑行样本进行测试
