# MTRRNet: Mamba + Transformer for Reflection Removal in Endoscopy Images
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from timm.models.vision_transformer import PatchEmbed
from timm.models.swin_transformer import SwinTransformerBlock
from timm.models.layers import DropPath
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2DLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, k_size, stride, padding=None, dilation=1, norm=None, act=None, bias=False):
        super(Conv2DLayer, self).__init__()  # super() 是一个内置函数，返回一个临时对象，该对象允许你调用父类中的方法。这里的 Conv2DLayer 是当前子类的名字。self 是当前实例对象
        # use default padding value or (kernel size // 2) * dilation value
        if padding is not None:
            padding = padding
        else:
            padding = dilation * (k_size - 1) // 2 # dilation指的是一种修改卷积操作的方法，它通过在卷积核中插入空洞（即在卷积核的元素之间增加间距）来扩大感受野
            # 用 add_module 方法将卷积层注册到模块中，命名为 'conv2d'，这样这个层可以在整个模型中被追踪和更新。
        self.add_module('conv2d', nn.Conv2d(in_channels, out_channels, k_size, stride, padding, dilation=dilation, bias=bias)) # k_size 通常是 kernel size（卷积核大小）的缩写
        if norm is not None: # "Norm" 归一化函数
            self.add_module('norm', norm(out_channels))
        if act is not None:
            self.add_module('act', act)

# 多尺度拉普拉斯特征提取
class LaplacianPyramid(nn.Module):
    # filter laplacian LOG kernel, kernel size: 3.
    # The laplacian Pyramid is used to generate high frequency images.

    def __init__(self, device='cuda', dim=3):
        super(LaplacianPyramid, self).__init__()

        # 2D laplacian kernel (2D LOG operator).
        self.channel_dim = dim
        laplacian_kernel = np.array([[0, -1, 0],[-1, 4, -1],[0, -1, 0]])

        laplacian_kernel = np.repeat(laplacian_kernel[None, None, :, :], dim, 0) # 变成 (dim, 1, H, W) 
        # learnable laplacian kernel


        # 让 kernel 可学习但只允许在微小范围内变动
        self.kernel = torch.nn.Parameter(torch.FloatTensor(laplacian_kernel))
        self.register_buffer('kernel_init', torch.FloatTensor(laplacian_kernel).clone())

        # 限制 kernel 在初始值±epsilon范围内
        epsilon = 0.05
        with torch.no_grad():
            self.kernel.data.clamp_(self.kernel_init - epsilon, self.kernel_init + epsilon)


    def forward(self, x):
        # print(self.kernel[0,0,:,:])
        # pyramid module for 4 scales.
        x0 = F.interpolate(x, scale_factor=0.125, mode='bilinear')# 下采样到 1/8
        x1 = F.interpolate(x, scale_factor=0.25, mode='bilinear')
        x2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        # groups=self.channel_dim：表示使用分组卷积，分组数为 self.channel_dim。当 groups 等于输入通道数时，相当于对每个通道进行独立卷积。
        lap_0 = F.conv2d(x0, self.kernel, groups=self.channel_dim, padding=1, stride=1, dilation=1)
        lap_1 = F.conv2d(x1, self.kernel, groups=self.channel_dim, padding=1, stride=1, dilation=1)
        lap_2 = F.conv2d(x2, self.kernel, groups=self.channel_dim, padding=1, stride=1, dilation=1)
        lap_3 = F.conv2d(x, self.kernel, groups=self.channel_dim, padding=1, stride=1, dilation=1)
        lap_0 = F.interpolate(lap_0, scale_factor=8, mode='bilinear')
        lap_1 = F.interpolate(lap_1, scale_factor=4, mode='bilinear')
        lap_2 = F.interpolate(lap_2, scale_factor=2, mode='bilinear')
        # lap_0, lap_1, lap_2, lap_3 是经过不同尺度的拉普拉斯卷积和插值后的图像高频部分。
        # 这些高频部分能够捕捉图像中的细节和边缘信息，有助于图像增强、复原等任务。
        # 最终的实现应包括将这些高频部分组合起来，以便进一步处理或计算损失

        return torch.cat([lap_0, lap_1, lap_2, lap_3], 1) # 使用 torch.cat 函数沿着指定的维度（在本例中为维度1，即通道维度）将多个张量拼接在一起
        # 返回(B,6*4,256,256)

class ChannelAttention(nn.Module):
    # The channel attention block
    # Original relize of CBAM module.
    # Sigma(MLP(F_max^c) + MLP(F_avg^c)) -> output channel attention feature.
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc_1 = nn.Conv2d(channel, channel // reduction, 1, bias=True)
        self.relu = nn.ReLU(True)
        self.fc_2 = nn.Conv2d(channel // reduction, channel, 1, bias=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_output = self.fc_2(self.relu(self.fc_1(self.avg_pool(x))))
        max_output = self.fc_2(self.relu(self.fc_1(self.max_pool(x))))
        out = avg_output + max_output
        return self.sigmoid(out)

# 对特征图的每个空间位置（像素）分配一个权重（0~1），突出重要区域并抑制无关背景。
class SpatialAttention(nn.Module):
    # The spatial attention block.
    # Simgoid(conv([F_max^s; F_avg^s])) -> output spatial attention feature.
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in [3, 7], 'kernel size must be 3 or 7.'
        padding_size = 1 if kernel_size == 3 else 3

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, padding=padding_size, bias=False, kernel_size=kernel_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True) # [B,1,H,W]
        max_out, _ = torch.max(x, dim=1, keepdim=True) # [B,1,H,W]

        pool_out = torch.cat([avg_out, max_out], dim=1) # [B,2,H,W]
        x = self.conv(pool_out) # 融合
        return self.sigmoid(x) # 输出

# 通道注意力+空间注意力
class CBAMlayer(nn.Module):
    # THe CBAM module(Channel & Spatial Attention feature) implement
    # reference from paper: CBAM(Convolutional Block Attention Module)
    def __init__(self, channel, reduction=16):
        super(CBAMlayer, self).__init__()
        self.channel_layer = ChannelAttention(channel, reduction)
        self.spatial_layer = SpatialAttention()

    def forward(self, x):
        x = self.channel_layer(x) * x
        x = self.spatial_layer(x) * x
        return x

# 带有通道注意力和空间注意力的残差快
class ResidualCbamBlock(nn.Module):
    # The ResBlock which contain CBAM attention module.

    def __init__(self, channel, norm=nn.BatchNorm2d, dilation=1, bias=False, act=nn.ReLU(True)):
        super(ResidualCbamBlock, self).__init__()

        self.conv1 = Conv2DLayer(channel, channel, k_size=3, stride=1, dilation=dilation, norm=norm, act=act, bias=bias)
        self.conv2 = Conv2DLayer(channel, channel, k_size=3, stride=1, dilation=dilation, norm=norm, act=None, bias=None)
        self.cbam_layer = CBAMlayer(channel)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.cbam_layer(x)

        out = x + res
        return out

class SElayer(nn.Module):
    # The SE_layer(Channel Attention.) implement, reference to:
    # Squeeze-and-Excitation Networks
    def __init__(self, channel, reduction=16):
        super(SElayer, self).__init__()
        # 即使原始输入特征图非常大，经过这个池化层之后，输出的特征图的每个通道都会缩小成一个单一的值。
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 自适应池化（Adaptive Pooling）是一种特殊的池化操作，可以将输入特征图缩放到指定的输出尺寸。参数 1 指定了输出特征图的高度和宽度均为 1，即将输入特征图缩放到 1x1 的大小
        self.se = nn.Sequential( # Sequential把括号内所有层打包
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True), # nn.ReLU(inplace=True) 是 PyTorch 中定义 ReLU 激活函数的一种方式，其中 inplace=True 表示在原地进行操作。这意味着激活函数将直接修改输入张量，而不创建新的张量。这可以节省内存，但需要注意，它会覆盖输入张量的值
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        # "Linear" 在深度学习中一般指线性层或全连接层，其作用是对输入数据进行仿射变换：
        # 即计算 Y = XWᵀ + b，其中 W 是权重矩阵，b 是偏置向量。

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)
        return x * y
        # 这里的 x 是输入特征图，y 是经过 Squeeze-and-Excitation 模块处理后的特征图。最终的输出是输入特征图和经过注意力机制处理后的特征图的逐元素相乘。

class SEResidualBlock(nn.Module):
    # The ResBlock implements: the conv & skip connections here.
    # Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf. 
    # Which contains SE-layer implements.
    def __init__(self, channel, norm=nn.BatchNorm2d, dilation=1, bias=False, se_reduction=None, res_scale=0.1, act=nn.GELU()):# 调用时既没有归一化 也没有激活
        super(SEResidualBlock, self).__init__()

        self.conv1 = Conv2DLayer(channel, channel, k_size=3, stride=1, dilation=dilation, norm=norm, act=act, bias=bias)
        self.conv2 = Conv2DLayer(channel, channel, k_size=3, stride=1, dilation=dilation, norm=norm, act=None, bias=None)
        self.se_layer = None
        self.res_scale = res_scale # res_scale 是一个缩放因子，用于对残差块的输出进行缩放。其主要目的是在训练过程中稳定网络的梯度，从而加速收敛并提高训练的稳定性。
        if se_reduction is not None: # se_reduction 通常与 Squeeze-and-Excitation (SE) 模块有关。SE 模块是一种在卷积神经网络（CNN）中的注意力机制，它通过自适应地重新校准通道特征来提升模型的表现。se_reduction 是 SE 模块中的一个参数，用于控制特征图在 Squeeze 阶段的通道缩减比例。
            self.se_layer = SElayer(channel, se_reduction)

    def forward(self, x):
        res = x # 残差
        x = self.conv1(x)
        x = self.conv2(x)
        if self.se_layer:
            x = self.se_layer(x) # 通道注意力
        x = x * self.res_scale 
        out = x + res # 残差链接
        return out

# --------------------------
# 编码块 CSA 
# --------------------------
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.csa0 = ResidualCbamBlock(channel=out_channels)
        self.csa1 = ResidualCbamBlock(channel=out_channels)
        self.out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),  # 添加 BatchNorm2d
            nn.GELU()                    
        )

    def forward(self, x):

        x = self.conv(x)
        x = self.csa0(x)
        x = self.csa1(x)
        return self.out(x)

# --------------------------
# 解码块
# --------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),  # 添加 BatchNorm2d
            nn.GELU()                    
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        return self.conv(x)

# --------------------------
# RDM 模块：完整结构版
# --------------------------
class RDM(nn.Module):
    def __init__(self):
        super().__init__()

        self.Lap = LaplacianPyramid(dim=3)
        self.se0 = SEResidualBlock(channel=6, se_reduction=2, res_scale=0.1)
        self.se1 = SEResidualBlock(channel=6, se_reduction=2, res_scale=0.1)
        self.se2 = SEResidualBlock(channel=6, se_reduction=2, res_scale=0.1)

        self.se3 = SEResidualBlock(channel=21, se_reduction=7, res_scale=0.1)
        self.se4 = SEResidualBlock(channel=21, se_reduction=7, res_scale=0.1)
        self.se5 = SEResidualBlock(channel=21, se_reduction=7, res_scale=0.1)

        # Encoder
        self.enc1 = EncoderBlock(21, 64)     # Conv + CSA + Res
        self.enc2 = EncoderBlock(64, 128)                 # Conv + CSA + Res
        self.enc3 = EncoderBlock(128, 256)                # Conv + CSA + Res

        # 空洞卷积
        self.diconv1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 2, dilation = 2),# dilation=2相当于kernel为5*5 扩大卷积核的感受野而不增加参数数量
            nn.BatchNorm2d(256), 
            nn.GELU() 
            )
        self.diconv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 4, dilation = 4),
            nn.BatchNorm2d(256), 
            nn.GELU() 
            )
        self.diconv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 8, dilation = 8),
            nn.BatchNorm2d(256), 
            nn.GELU() 
            )
        self.diconv4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 16, dilation = 16),
            nn.BatchNorm2d(256), 
            nn.GELU() 
            )

        # Decoder
        self.dec1 = DecoderBlock(256, 128)  # 修改in_channels = 384
        self.dec2 = DecoderBlock(128, 64)    # 修改in_channels = 192
        self.dec3 = DecoderBlock(64, 32)          # 无跳跃连接，保持不变

        # Output
        self.out_head = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()          
        )
        

    def forward(self, x):
        res = x
        lap = self.Lap(x)
        x = torch.cat([x, x],dim=1) # B 6 256 256
        x = self.se0(x)
        x = self.se1(x)
        x = self.se2(x)
        x = torch.cat([res, x, lap], dim=1) # B 21 256 256
        x = self.se3(x)
        x = self.se4(x)
        x = self.se5(x)


        x = self.enc1(x)       # (B,64,128,128)
        res1 = x
        x = self.enc2(x)      # (B,128,64,64)
        res2 = x
        x = self.enc3(x)      # (B,256,32,32)

        x = self.diconv1(x) # # (B,256,32,32)
        x = self.diconv2(x)
        x = self.diconv3(x)
        x = self.diconv4(x)
        
        x = self.dec1(x) + res2      # (B,128,64,64)
        x = self.dec2(x) + res1      # (B,64,128,128)
        x = self.dec3(x)      # (B,32,256,256)

        out = self.out_head(x) # (B,1,256,256)
        return out


# --------------------------
# Patch Embedding
# --------------------------
class ConvPatchEmbed(nn.Module):
    def __init__(self, in_chs, embed_dim, kernel_size=4, stride=4):
        super().__init__()
        # 使用卷积代替分块编码，非重叠分块+升维
        self.proj = nn.Sequential(
            nn.Conv2d(in_chs, embed_dim, kernel_size, stride, kernel_size // 2),
            nn.BatchNorm2d(embed_dim),  # 添加 BatchNorm2d
            nn.GELU()                 # 添加 PReLU 激活
        )

    def forward(self, x):
        return self.proj(x)



# --------------------------
# Mamba2D Block
# --------------------------
class MambaBlock2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)  # 通道维归一化（用于序列）
        self.mamba = Mamba(dim)          # 原始 Mamba 是处理序列的
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),         # 添加归一化层以稳定激活和梯度
            nn.GELU()
        )


    def forward(self, x):
        """
        输入 x: [B, C, H, W]
        输出:   [B, C, H, W]
        """
        x = self.norm(x)
        x = torch.clamp(x, -10.0, 10.0)  # 限制极值
        res = x
        B, C, H, W = x.shape
        x_perm = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, C, H, W] -> [B, H*W, C]

        with torch.cuda.amp.autocast(enabled=False):
            x_seq = self.mamba(x_perm)
        x_out = x_seq.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, H*W, C] -> [B, C, H, W]

        # 修复 scale_raw 的范围问题

        res_out = res + self.proj(x_out)

        # 限制输出范围
        res_out = torch.clamp(res_out, min=-10.0, max=10.0)

        return res_out


# --------------------------
# Swin Transformer
# --------------------------
class ResidualSwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size, shift_size, mlp_ratio):
        super().__init__()
        self.Swin = SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
                mlp_ratio=mlp_ratio
            )
        self.norm = nn.GroupNorm(1,dim)  

    def forward(self, x):
        '''
        输入 x:[B,C,H,W]
        输出 x:[B,C,H,W]
        '''
        x = self.norm(x)             # 归一化
        x = x.permute(0, 2, 3, 1)
        residual = x
        x = self.Swin(x) + residual  # 残差连接
        x = x.permute(0, 3, 1, 2)
        x = torch.clamp(x, min=-10.0, max=10.0)
        return x

# --------------------------
# Swin Transformer + Mamba Block
# --------------------------
class SwinMambaBlock(nn.Module):
    def __init__(self, dim, input_resolution, swin_blocks=2, mamba_blocks=2):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.norm_act1 = nn.Sequential(
            nn.GroupNorm(1,dim),  
            nn.GELU()             # 添加 PReLU 激活
        )
        self.swin = nn.Sequential(*[
            ResidualSwinBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=dim // 32,
                window_size=7,
                shift_size=3,
                mlp_ratio=4.0
            ) for _ in range(swin_blocks)
        ])
        self.mamba = nn.Sequential(*[MambaBlock2D(dim) for _ in range(mamba_blocks)])
        self.norm_act2 = nn.Sequential(
            nn.GroupNorm(1,dim),  
            nn.GELU()             # 添加 PReLU 激活
        )


    def forward(self, x):

        x = self.norm_act1(x)
        res1 = x
        # 进入 Swin 层
        x = self.swin(x)  # 输出仍是 [B, H, W, C]
        x = x + res1
        x = torch.clamp(x, min=-10.0, max=10.0)

        x = self.norm_act2(x)
        res2 = x
        x = self.mamba(x)                         # 输入 Mamba
        x = x + res2
        x = torch.clamp(x, min=-10.0, max=10.0)

        return x


# --------------------------
# SubNet 多尺度融合模块
# --------------------------
# 修改SubNet中的fuse操作
class SubNet(nn.Module):
    def __init__(self, in_dims=(64, 128, 256), num_layers=(1, 1, 1)):
        super().__init__()

        # Level 0: 接收 c1 上采样特征
        self.fuse0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_dims[1], in_dims[0], kernel_size=1),
            nn.BatchNorm2d(in_dims[0]),  # 添加BatchNorm
            nn.GELU()  # 使用PReLU代替线性激活
        )
        self.m0 = nn.Sequential(*[MambaBlock2D(in_dims[0]) for _ in range(num_layers[0])])

        # Level 1: 融合 down(c0) + up(c2)
        self.down_c0 = nn.Sequential(
            nn.Conv2d(in_dims[0], in_dims[1], kernel_size=3, stride=2, padding=1)
        )
        self.up_c2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_dims[2], in_dims[1], kernel_size=1)
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(in_dims[1], in_dims[1], kernel_size=1),
            nn.BatchNorm2d(in_dims[1]),  # 添加BatchNorm
            nn.GELU()  # 使用PReLU代替线性激活
        )
        self.m1 = nn.Sequential(*[MambaBlock2D(in_dims[1]) for _ in range(num_layers[1])])

        # Level 2: 接收 c1 下采样特征
        self.down_c1 = nn.Sequential(
            nn.Conv2d(in_dims[1], in_dims[2], kernel_size=3, stride=2, padding=1)
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(in_dims[2], in_dims[2], kernel_size=1),
            nn.BatchNorm2d(in_dims[2]),  # 添加BatchNorm
            nn.GELU()  # 使用PReLU代替线性激活
        )
        self.m2 = nn.Sequential(*[MambaBlock2D(in_dims[2]) for _ in range(num_layers[2])])
        
        shortcut_scale_init_value = 0.5
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, in_dims[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, in_dims[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, in_dims[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
    
    def safe_add(self, x, y):
        # 相同的实现
        if x.shape != y.shape:
            y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
        return x + y

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign() # ​​符号保留​​
            data.abs_().clamp_(value) # 将输入张量 data 的每个元素的绝对值限制在 [value, +∞) 范围内
            data *= sign
        
    def forward(self, c0, c1, c2):
        self._clamp_abs(self.alpha0.data, 1e-3) 
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        
        # Level 0: 上采样c1融合
        feat0 = self.fuse0(c1)
        feat0 = torch.clamp(feat0, -5.0, 5.0)  # 添加clamp
        c0_out = self.m0(feat0)
        c0 = self.safe_add(self.alpha0*c0, (1-self.alpha0)*c0_out)
        c0 = torch.clamp(c0, -5.0, 5.0)  # 添加clamp
        
        # Level 1: down(c0) + up(c2)
        feat1 = self.down_c0(c0) + self.up_c2(c2)
        feat1 = self.fuse1(feat1)
        feat1 = torch.clamp(feat1, -5.0, 5.0)  # 添加clamp
        c1_out = self.m1(feat1)
        c1 = self.safe_add(self.alpha1*c1, (1-self.alpha1)*c1_out)
        c1 = torch.clamp(c1, -5.0, 5.0)  # 添加clamp
        
        # Level 2: down(c1)
        feat2 = self.down_c1(c1)
        feat2 = self.fuse2(feat2)
        feat2 = torch.clamp(feat2, -5.0, 5.0)  # 添加clamp
        c2_out = self.m2(feat2)
        c2 = self.safe_add(self.alpha2*c2, (1-self.alpha2)*c2_out)
        c2 = torch.clamp(c2, -5.0, 5.0)  # 添加clamp
        
        return c0, c1, c2


# --------------------------
# 解码器
# --------------------------
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 两级上采样并插入 Mamba
        self.up1 = nn.ConvTranspose2d(256,128,2,2)
        self.m1 = MambaBlock2D(128)
        self.up2 = nn.ConvTranspose2d(128,64,2,2)
        self.m2 = MambaBlock2D(64)
        self.out_conv = nn.Sequential(
            nn.Conv2d(64,3,3,1,1),  # 输出 RGB 图 前3通道为去反光图，后3通道为反光预测图
            nn.BatchNorm2d(3),  # 添加 BatchNorm2d
            nn.Sigmoid()          # 添加 PReLU 激活
        )

    def forward(self, c0, c1, c2):
        x = self.up1(c2)
        if x.shape[-2:] != c1.shape[-2:]:
            x = F.interpolate(x, size=c1.shape[-2:], mode='bilinear', align_corners=False)
        x = self.m1(x + c1)

        x = self.up2(x)
        if x.shape[-2:] != c0.shape[-2:]:
            x = F.interpolate(x, size=c0.shape[-2:], mode='bilinear', align_corners=False)        
        x = self.m2(x + c0)

        x = self.out_conv(x)

        # 添加上采样，将 x 恢复为与输入相同的尺寸
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        return x

# --------------------------
# 主模型
# --------------------------
class MTRRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rdm = RDM()  # 提取高光区域图

        # 初始Patch Embedding：x_in + rmap => 4通道 => 192维编码
        self.patch_embed0 = ConvPatchEmbed(4, 192, kernel_size=4, stride=4)

        # 编码器三阶段：Swin+Mamba 堆叠结构
        self.encoder0 = nn.Sequential(*[SwinMambaBlock(192, input_resolution=(64,64)) for _ in range(2)])
        self.patch_embed1 = ConvPatchEmbed(192, 384, 2, 2)
        self.encoder1 = nn.Sequential(*[SwinMambaBlock(384, input_resolution=(32,32)) for _ in range(3)])
        self.patch_embed2 = ConvPatchEmbed(384, 768, 2, 2)
        self.encoder2 = nn.Sequential(*[SwinMambaBlock(768, input_resolution=(16,16), mamba_blocks=3) for _ in range(4)])

        # 特征通道适配
        self.c0_adapter = nn.Conv2d(192, 64, 1)
        self.c1_adapter = nn.Conv2d(384,128,1)
        self.c2_adapter = nn.Conv2d(768,256,1)

        # 三层 SubNet，不共享参数
        self.subnet0 = SubNet()
        self.subnet1 = SubNet()
        self.subnet2 = SubNet()

        # 解码器
        self.decoder1 = Decoder()
        self.decoder2 = Decoder()

        self.out1 = nn.Sequential(
            nn.BatchNorm2d(3),  # 添加 BatchNorm2d
            nn.Sigmoid()          # 添加 PReLU 激活
        )        
        self.out2 = nn.Sequential(
            nn.BatchNorm2d(3),  # 添加 BatchNorm2d
            nn.Sigmoid()          # 添加 PReLU 激活
        )        

    def make_even_hw(self,tensor):# 偶数化高度和宽度
        h, w = tensor.shape[-2], tensor.shape[-1]
        new_h = h if h % 2 == 0 else h - 1
        new_w = w if w % 2 == 0 else w - 1
        if new_h != h or new_w != w:
            tensor = tensor[..., :new_h, :new_w]
        return tensor

    def forward(self, x_in):
        rmap = self.rdm(x_in)  # 提取反光区域图
        x = torch.cat([x_in, rmap], dim=1)  # 拼接形成4通道输入

        x0 = self.make_even_hw(self.patch_embed0(x))  # 分块编码
        # print('x0.shape', x0.shape)
        x0 = self.encoder0(x0)

        x1 = self.make_even_hw(self.patch_embed1(x0))
        # print('x1.shape', x1.shape)
        x1 = self.encoder1(x1)

        x2 = self.make_even_hw(self.patch_embed2(x1))
        # print('x2.shape', x2.shape)
        x2 = self.encoder2(x2)

        # 通道转换
        c0 = self.make_even_hw(self.c0_adapter(x0))
        c1 = self.make_even_hw(self.c1_adapter(x1))
        c2 = self.make_even_hw(self.c2_adapter(x2))

        # print('c0.shape', c0.shape)
        # print('c1.shape', c1.shape)
        # print('c2.shape', c2.shape)

        # 三层子网络增强
        c0, c1, c2 = self.subnet0(c0, c1, c2)
        c0, c1, c2 = self.subnet1(c0, c1, c2)
        c0, c1, c2 = self.subnet2(c0, c1, c2)

        # 解码重建残差图
        clean = self.decoder1(c0, c1, c2)
        ref = self.decoder2(c0, c1, c2)

        return rmap, self.out1(x_in+ref), self.out2(x_in+clean)  



class MTRREngine(nn.Module):

    def __init__(self, opts, device):
        super(MTRREngine, self).__init__()
        self.device = device 
        self.opts  = opts
        self.visual_names = ['fake_T', 'fake_R', 'c_map', 'I', 'T', 'R']
        self.netG_T = MTRRNet().to(device) 
        self.netG_T.apply(self.init_weights)


    def load_checkpoint(self, optimizer):
        if self.opts.model_path is not None:
            checkpoint_path = self.opts.model_path
            print('Load the model from %s' % checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=str(self.device))
            
            # 兼容只包含model_state_dict的情况
            if isinstance(checkpoint, dict) and 'model_state_dict' not in checkpoint :
                self.netG_T.load_state_dict({k.replace('netG_T.', ''): v for k, v in checkpoint.items()})
                return None
            else:
                self.netG_T.load_state_dict({k.replace('netG_T.', ''): v for k, v in checkpoint['model_state_dict'].items()})

                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = checkpoint['lr']

                epoch = checkpoint.get('epoch', None)
            return epoch
        

    def get_current_visuals(self):
        # get the current visuals results.
        visual_result = OrderedDict() # 是 Python 标准库 collections 模块中的一个类，它是一个有序的字典，记录了键值对插入的顺序。
        for name in self.visual_names: # 这里遍历 self.visual_names 列表，该列表包含了需要获取的属性名称。 ['fake_Ts', 'fake_Rs', 'rcmaps', 'I'] 都在本class有定义
            if isinstance(name, str): # 检查 name 是否是字符串
                # 使用 getattr(self, name) 函数动态地获取 self 对象中名为 name 的属性的值，并将其存储在 visual_result 字典中
                visual_result[name] = getattr(self, name)
        return visual_result # 结果从 visual_names 来


    def set_input(self, input): 
        # load images dataset from dataloader.
        self.I = input['input'].to(self.device)
        self.T = input['target_t'].to(self.device)
        self.R = input['target_r'].to(self.device)
        


    def forward(self):
        # self.init()
        self.c_map, self.fake_R, self.fake_T = self.netG_T(self.I) 

        
 
    def monitor_layer_stats(self):
        """仅监控模型的一级子模块（不深入嵌套结构）"""
        hooks = []
        model = self.netG_T

        # 修正钩子函数参数（正确接收module, input, output）
        def _hook_fn(module, input, output, layer_name):
            if isinstance(output, torch.Tensor):
                mean = output.mean().item()
                std = output.std().item()
                msg = f"{layer_name:<50} | Mean: {mean:>15.6f} | Std: {std:>15.6f} | Shape: {tuple(output.shape)}"
                print(msg)
                with open('./state.log', 'a') as f:
                    f.write(msg + '\n')# 修正钩子函数参数（正确接收module, input, output）
      

        # 遍历所有子模块并注册钩子
        for name, module in model.named_modules():
            if not isinstance(module, nn.ModuleList):  # 过滤容器类（如Sequential）
                hook = module.register_forward_hook(
                    lambda m, inp, out, name=name: _hook_fn(m, inp, out, name)
                )
                hooks.append(hook)   

        # 执行前向传播
        with torch.no_grad():
            self.forward()
        
        # 移除钩子
        for hook in hooks:
            hook.remove()

    def monitor_layer_grad(self):
        with open('./grad.log', 'a') as f:
            for name, param in self.netG_T.named_parameters():
                if param.grad is not None:
                    msg = (
                        f"Param: {name:<50} | "
                        f"Grad Mean: {param.grad.mean().item():.15f} | "
                        f"Grad Std: {param.grad.std().item():.15f}"
                    )
                else:
                    msg = f"Param: {name:<50} | Grad is None"  # 梯度未回传  
                print(msg)
                f.write(msg + '\n')

    def eval(self):
        self.netG_T.eval()

    def inference(self):
        # with torch.no_grad():
        self.forward()             #所以启动全部模型的最高层调用

    def count_parameters(self):
        table = []
        total = 0
        for name, param in self.netG_T.named_parameters():
            if param.requires_grad:
                num = param.numel()
                table.append([name, num, f"{num:,}"])
                total += num
        print(tabulate(table, headers=["Layer", "Size", "Formatted"], tablefmt="grid"))
        print(f"\nTotal trainable parameters: {total:,}")    

    def apply_weight_constraints(self):
        """动态裁剪权重，保持在合理范围内"""
        with torch.no_grad():
            for name, param in self.netG_T.named_parameters():
                # 针对不同参数类型使用不同的裁剪策略
                
                # PReLU参数特殊约束，避免负斜率过大
                if any(x in name for x in ['proj.2.weight', '.out.2.weight']) or ('norm_act' in name and name.endswith('.weight')):
                    param.data.clamp_(min=0.01, max=0.3)
                    
                # scale_raw参数约束
                elif name.endswith('scale_raw'):
                    param.data.clamp_(min=-2.0, max=2.0)
                    
                # 普通权重参数通用约束
                elif 'weight' in name and param.dim() > 1:
                    if param.numel() == 0:
                        continue  # 跳过空张量
                    if torch.max(torch.abs(param.data)) > 10.0:
                        param.data.clamp_(min=-10.0, max=10.0)

    @staticmethod
    def init_weights(m):
        # 通用卷积层
        if isinstance(m, (nn.Conv2d, nn.Conv1d)):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # 通用线性层
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # LayerNorm和BatchNorm
        elif isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # PReLU特殊初始化 - 避免梯度不稳定
        elif isinstance(m, nn.PReLU):
            # 使用保守的小正值初始化PReLU参数，避免过大的负斜率
            nn.init.uniform_(m.weight, 0.05, 0.1)

        # 针对自定义模块/参数名
        for name, param in m.named_parameters(recurse=False):
            # 常见proj和自定义权重
            if any([k in name.lower() for k in ['proj', 'out_proj', 'x_proj', 'conv', 'weight']]):
                if param.dim() >= 2:  # 只初始化权重，不初始化bias
                    # 用xavier对proj类参数更稳妥
                    nn.init.xavier_uniform_(param)
                elif param.dim() == 1:  # bias或者norm的weight
                    if 'bias' in name or 'beta' in name:
                        nn.init.zeros_(param)
                    elif 'weight' in name or 'gamma' in name:
                        nn.init.ones_(param)
            
            
    

# --------------------------
# 模型验证
# --------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MTRRNet().to(device)
    x = torch.randn(1,3,256,256).to(device)  # 输入一张256x256 RGB图
    y = model(x)
    print(y.shape)  # 应输出 torch.Size([1, 3, 256, 256])
