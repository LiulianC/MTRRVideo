# MTRRNet Token-only Architecture

## 概述

本次更新将MTRRNet从"每层局部解码 + 图像域多次往返"改造为"Token-only 多尺度编码 → Token-only SubNet 融合 → 统一 Decoder 一次性从 Token 解码为 6 通道 (T,R)"的全新架构。

## 新架构特点

### 1. 多尺度Token编码 (MultiScaleTokenEncoder)
- **4个尺度**: 256×256, 128×128, 64×64, 32×32 (对应原encoder0~3)
- **频带分工**: 
  - 低频 (Gaussian blur) → Mamba stack 处理
  - 高频 (input - blur) → Swin stack 处理
- **逐层残差**: 所有block采用 `x = x + f(LN(x))` 形式
- **Token输出**: 每个尺度输出 token (B, N_i, C_i)

### 2. 频带分离 (FrequencySplit)
```python
# 可学习的高斯模糊分离低频/高频
low_freq, high_freq = freq_split(x)
# low_freq → Mamba (长序列建模)
# high_freq → Swin (局部细节处理)
```

### 3. Token SubNet融合 (TokenSubNet)
- **对齐策略**: 所有尺度token插值到统一64×64网格
- **融合方式**: 通道concat后1×1卷积融合
- **细化处理**: 轻量Mamba blocks进行token交互

### 4. 统一解码器 (UnifiedTokenDecoder)
- **Token→Feature**: Linear投影到feature map
- **上采样**: ConvTranspose2d 64×64 → 256×256
- **Base缩放**: `output = base_scale * input + delta`
- **6通道输出**: 前3通道fake_T，后3通道fake_R

### 5. 中间监督
- **每尺度**: 轻量aux_head生成低分辨率预测
- **可视化**: 缓存到`self.intermediates`字典
- **不影响训练**: detach方式，训练脚本可选择性使用

## 使用方式

### 基本用法
```python
from MTRRNet import MTRRNet

# 新Token-only架构 (默认)
model = MTRRNet(use_legacy=False)

# 向后兼容的Legacy架构
model_legacy = MTRRNet(use_legacy=True)
```

### 中间监督可视化
```python
rmap, out = model(x)
intermediates = model.get_intermediates()
# intermediates = {'aux_s0': pred0, 'aux_s1': pred1, ...}
```

### Token统计监控
```python
debug_stats = model.get_debug_stats()
# debug_stats = {'tokens_s0_mean': val, 'fused_tokens_std': val, ...}
```

## 接口兼容性

### MTRREngine修改
- **net_c暂停**: 直接使用`self.I`而非`self.Ic`
- **接口保持**: `forward()`返回格式不变
- **监控增强**: 新增`monitor_token_stats()`

### 训练脚本
- **无需修改**: 现有train.py保持完全兼容
- **loss计算**: 输出格式与原版一致 (6通道)
- **可选功能**: 可读取`model.get_intermediates()`进行额外监督

## 核心模块说明

### token_modules.py
- `FrequencySplit`: 频带分离 (可学习高斯核)
- `TokenPatchEmbed`: 图像patch → token嵌入
- `MambaTokenBlock`: Mamba处理token序列
- `SwinTokenBlock`: Swin处理token网格  
- `TokenStage`: 单尺度完整处理流程
- `MultiScaleTokenEncoder`: 多尺度token编码
- `TokenSubNet`: 多尺度token融合
- `UnifiedTokenDecoder`: token统一解码

### 关键参数
- `embed_dims`: [192, 192, 96, 96] (各尺度token维度)
- `ref_resolution`: 64 (SubNet统一分辨率)
- `base_scale_init`: 0.3 (residual base缩放因子)
- `mamba_blocks`: [10, 10, 10, 10] (各尺度Mamba深度)
- `swin_blocks`: [4, 4, 4, 4] (各尺度Swin深度)

## 监控与调试

### Token统计
```bash
# 训练时开启debug模式
opts.debug_monitor_layer_stats = 1

# 查看token统计日志
tail -f ./debug/token_stats.log
```

### 中间可视化
研究者可在训练脚本中添加:
```python
if epoch % 10 == 0:  # 每10轮可视化一次
    intermediates = model.netG_T.get_intermediates()
    for name, pred in intermediates.items():
        save_image(pred, f'./vis/{name}_epoch{epoch}.png')
```

## 性能优化

### 内存效率
- Token表示相比feature map更紧凑
- 统一解码避免多次上采样
- 梯度检查点可继续使用

### 计算效率  
- Mamba线性复杂度处理长序列
- Swin window attention减少计算量
- 单次解码替代多层往返

## 测试验证

运行测试脚本验证新架构:
```bash
python test_token_architecture.py
```

测试内容:
- [x] Token模块导入和实例化
- [x] MTRRNet新旧模式切换
- [x] 前向传播形状检查 (需PyTorch环境)
- [x] 中间监督和debug统计