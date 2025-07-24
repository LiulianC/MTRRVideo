MTRRNet ASCII 风格 前向传播 网络图
│
├── Input: x_in (B,3,256,256)
│
├── RDM: x_in (B,3,256,256)                             # 输入RGB图像
│   ├── Encoder                                         # 编码器阶段：提取深层反射特征
│   │   ├── LapConv + CSA + ResBlk ──► (B,64,128,128)    # 下采样 ×2，提取高频+注意力+残差 Lap算子是在一定范围内可学习的！
│   │   ├── Conv + CSA + ResBlk ──► (B,128,64,64)       # 下采样 ×2
│   │   └── Conv + CSA + ResBlk ──► (B,256,32,32)       # 下采样 ×2，最深特征
│   │
│   ├── Decoder                                         # 解码器阶段：还原空间尺寸
│   │   ├── Upsample + Skip + Conv ──► (B,128,64,64)      # 上采样 + 跳跃连接
│   │   ├── Upsample + Skip + Conv ──► (B,64,128,128)     # 上采样 + 跳跃连接
│   │   └── Upsample + Conv ─────────► (B,32,256,256)     # 最后一层上采样
│   │
│   └── Output Head: Conv 1×1 ───────► Rmap (B,1,256,256) # 输出反射区域图（高光概率图）
│
├── cat([x_in, Rmap], dim=1) ───► x (B,4,256,256)
│
├── Stem Processing
│   ├── Initial PatchEmbed ───► x_emb (B,192,64,64)        # 4×4 Conv, stride=4
│   │
│   ├── Stage0 PatchEmbed     ───► x0 (B,192,64,64)        # 恒定，不变分辨率
│   └── Stage0 Encoder    
│       └── (SwinTransformer × 3 + Mamba × 2) × 2 ───► c0_raw (B,192,64,64)
│
│   ├── Stage1 PatchEmbed     ───► x1 (B,384,32,32)        # PatchEmbed: 2×2 Conv, stride=2
│   └── Stage1 Encoder 
│       └── (SwinTransformer × 3 + Mamba × 2) × 3 ───► c1_raw (B,384,32,32)
│
│   ├── Stage2 PatchEmbed     ───► x2 (B,768,16,16)        # PatchEmbed: 2×2 Conv, stride=2
│   └── Stage2 Encoder 
│       └── (SwinTransformer × 2 + Mamba × 2) × 4 ───► c2_raw (B,768,16,16)
│
├── channels_adapter: 1×1 Conv
│   ├── c0_raw ───► c0 (B, 64,64,64)
│   ├── c1_raw ───► c1 (B,128,32,32)
│   └── c2_raw ───► c2 (B,256,16,16)
│
├── SubNet 循环 (3次，不共享权重) 
│   ├── SubNet0
│   │   ├── Level0
│   │   │   ├── Fusion0: up(c1→64) ───► feat0
│   │   │   └── MAMBA×L0 ───► level0_out
│   │   └── c0 = α0*c0 + level0_out
│   │
│   │   ├── Level1
│   │   │   ├── Fusion0: down(c0→128) + up(c2→128) ───► feat1
│   │   │   └── MAMBA×L1 ───► level1_out
│   │   └── c1 = α1*c1 + level1_out
│   │
│   │   ├── Level2
│   │   │   ├── Fusion0: down(c1→256) ───► feat2
│   │   │   └── MAMBA×L2 ───► level2_out
│   │   └── c2 = α2*c2 + level2_out
│   │
│   ├── SubNet1 (同上，输入为 SubNet0 输出)
│   └── SubNet2 (同上，输入为 SubNet1 输出)
│
├── Decoder
│   ├── Clean Path (x_clean 去反光图)
│   │   ├── c2_clean ──► Upsample (×2) ──► concat c1_clean
│   │   │                                ↓
│   │   │                        MAMBA Block ①
│   │   │                                ↓
│   │   ├── ──► Upsample (×2) ──► concat c0_clean
│   │   │                                ↓
│   │   │                        MAMBA Block ②
│   │   │                                ↓
│   │   └────────────────────────► Conv(3x3) ──► x_clean ∈ (B,3,256,256)
│   │
│   ├── Ref Path (x_ref 反光图)
│   │   ├── c2_ref ──► Upsample (×2) ──► concat c1_ref
│   │   │                              ↓
│   │   │                      MAMBA Block ①
│   │   │                              ↓
│   │   ├── ──► Upsample (×2) ──► concat c0_ref
│   │   │                              ↓
│   │   │                      MAMBA Block ②
│   │   │                              ↓
│   │   └──────────────────────► Conv(3x3) ──► x_ref ∈ (B,3,256,256)
│   │
│   └── Output 拼接
│       └── cat([x_clean, x_ref], dim=1) ──► x_out ∈ (B,6,256,256)
│
│
└── Outputs
    └── x_img_out = x_in - c_clean   (残差输出) 前三通道是去反光图 后三通道是反光图
     



我现在做内窥镜图像去除反光的任务，并做好图像后将其扩展到去除视频反光。总体思路是采用mamba与transformer的混合模型。具体思路是：一个256*256大小的RGB图像，经过一个能提取反射先验的，由拉普拉斯卷积和通道空间注意力残差连接块改进的Unet组成的能输出高光区域的模块，获得通道数为1的高光位置图。接着将高光位置图与原RGB图片分成多个不重叠块，通道数变为192。之后把图像下采样两次，分辨率最高的stage0采用(SwinTransformer × 3 + Mamba × 2) × 2，分辨率次高的stage1由(SwinTransformer × 3 + Mamba × 2) × 3 来提取特征，分辨率最低的stage2由(SwinTransformer × 2 + Mamba × 2) × 4 提取特征，得到c0_raw、c1_raw、c2_raw。把c0_raw、c1_raw、c2_raw进行通道数调整成c0\c1\c2。最后将c0、c1、c2打包，输入到一个包含三个子网络的多尺度特征的融合模块和一个解码模块，并残差连接最终的解码输出。

mamba使用from mamba_ssm import Mamba
swin transformer采用timm的库