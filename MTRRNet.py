# MTRRNet: Mamba + Transformer for Reflection Removal in Endoscopy Images
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from timm.models.vision_transformer import PatchEmbed
from timm.models.swin_transformer import SwinTransformerBlock
from timm.layers import DropPath
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from classifier import PretrainedConvNext_e2e
# from nafblock import NAFBlock
import torch.utils.checkpoint as checkpoint
import math
from timm.layers import LayerNorm2d


# padding是边缘复制 减少边框伪影
class Conv2DLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, norm=None, act=None, bias=False):
        super(Conv2DLayer, self).__init__()

        # Replication padding （复制边缘）
        if padding > 0:
            self.add_module('pad', nn.ReplicationPad2d(padding))  # [left, right, top, bottom] 都为 padding

        # 卷积
        self.add_module('conv2d', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation, groups=groups, bias=bias))

        # 归一化
        if norm is not None:
            self.add_module('norm', norm(out_channels))

        # 激活
        if act is not None:
            self.add_module('act', act)

# Attention-Aware Fusion
class AAF(nn.Module):
    """
    输入: List[Tensor], 每个 shape 为 [B, C, H, W]
    输出: Tensor, shape 为 [B, C, H, W]
    """
    def __init__(self, in_channels, num_inputs): # in_channels 每个图像的通道 num_input 有多少个图像
        super(AAF, self).__init__()
        self.in_channels = in_channels
        self.num_inputs = num_inputs
        
        # 输入 concat 后通道为 C*num_inputs
        self.attn = nn.Sequential(
            nn.Conv2d(num_inputs * in_channels, num_inputs * in_channels * 16, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_inputs * in_channels * 16, num_inputs, kernel_size=1, bias=False),
            nn.Softmax(dim=1)  # 对每个位置的 num_inputs 做归一化
        )

    def forward(self, features):
        # features: list of Tensors [B, C, H, W]
        B, C, H, W = features[0].shape
        x = torch.cat(features, dim=1)  # shape: [B, C*num_inputs, H, W]
        attn_weights = self.attn(x)     # shape: [B, num_inputs, H, W]
        
        # 融合：对每个尺度乘以权重后相加
        out = 0
        for i in range(self.num_inputs):
            weight = attn_weights[:, i:i+1, :, :]  # [B,1,H,W]
            out += features[i] * weight            # 广播乘法
        return out

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

        self.aaf = AAF(3,4)

        # self.conv0 = Conv2DLayer(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, bias=False, norm=nn.BatchNorm2d, act=nn.GELU())
        # self.conv1 = Conv2DLayer(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, bias=False, norm=nn.BatchNorm2d, act=nn.GELU())
        # self.conv2 = Conv2DLayer(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, bias=False, norm=nn.BatchNorm2d, act=nn.GELU())
        # self.conv3 = Conv2DLayer(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, bias=False, norm=nn.BatchNorm2d, act=nn.GELU())

    def forward(self, x):
        # print(self.kernel[0,0,:,:])
        # pyramid module for 4 scales.
        x0 = F.interpolate(x, scale_factor=0.125, mode='bicubic')# 下采样到 1/8
        x1 = F.interpolate(x, scale_factor=0.25, mode='bicubic')
        x2 = F.interpolate(x, scale_factor=0.5, mode='bicubic')
        # groups=self.channel_dim：表示使用分组卷积，分组数为 self.channel_dim。当 groups 等于输入通道数时，相当于对每个通道进行独立卷积。
        lap_0 = F.conv2d(x0, self.kernel, groups=self.channel_dim, padding=1, stride=1, dilation=1)
        lap_1 = F.conv2d(x1, self.kernel, groups=self.channel_dim, padding=1, stride=1, dilation=1)
        lap_2 = F.conv2d(x2, self.kernel, groups=self.channel_dim, padding=1, stride=1, dilation=1)
        lap_3 = F.conv2d(x, self.kernel, groups=self.channel_dim, padding=1, stride=1, dilation=1)
        lap_0 = F.interpolate(lap_0, scale_factor=8, mode='bicubic')
        lap_1 = F.interpolate(lap_1, scale_factor=4, mode='bicubic')
        lap_2 = F.interpolate(lap_2, scale_factor=2, mode='bicubic')
        # lap_0 =  self.conv0(lap_0)
        # lap_1 =  self.conv1(lap_1)
        # lap_2 =  self.conv2(lap_2)
        # lap_3 =  self.conv3(lap_3)

        lap_out = torch.cat([lap_0, lap_1, lap_2, lap_3],dim=1)

        return lap_out, x0,x1,x2 


class ChannelAttention(nn.Module):
    # The channel attention block
    # Original relize of CBAM module.
    # Sigma(MLP(F_max^c) + MLP(F_avg^c)) -> output channel attention feature.
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()

        # self.norm = nn.BatchNorm2d(channel)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 更稳定的初始化 + LayerNorm 替代 BatchNorm
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=True),
            nn.LayerNorm([channel * reduction, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel * reduction, channel, 1, bias=True),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # x = torch.clamp(x, -10.0, 10.0)  # 限制极值
        avg_output = self.fc(torch.tanh(self.avg_pool(x)) * 3)
        max_output = self.fc(torch.tanh(self.max_pool(x)) * 3)

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

        self.conv = Conv2DLayer(in_channels=2, out_channels=1, padding=padding_size, bias=False, kernel_size=kernel_size)
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
    def __init__(self, channel, reduction=1):
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

    def __init__(self, channel, reduction, norm=nn.BatchNorm2d, dilation=1, bias=False, act=nn.ReLU(True)):
        super(ResidualCbamBlock, self).__init__()

        self.conv1 = Conv2DLayer(channel, channel, kernel_size=3, stride=1, padding=1, dilation=dilation, norm=norm, act=act, bias=bias)
        self.conv2 = Conv2DLayer(channel, channel, kernel_size=3, stride=1, padding=1, dilation=dilation, norm=norm, act=None, bias=None)
        self.cbam_layer = CBAMlayer(channel,reduction=1)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.cbam_layer(x)

        out = x + res
        return out

class SElayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SElayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.se = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = ((torch.tanh(self.se(y))+1)/2).view(b, c, 1, 1)
        return x * y

class SEResidualBlock(nn.Module):
    # The ResBlock implements: the conv & skip connections here.
    # Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf. 
    # Which contains SE-layer implements.
    def __init__(self, channel, norm=nn.BatchNorm2d, dilation=1, bias=False, se_reduction=None, res_scale=0.1, act=nn.GELU()):# 调用时既没有归一化 也没有激活
        super(SEResidualBlock, self).__init__()

        self.conv1 = Conv2DLayer(channel, channel, kernel_size=3, stride=1, padding=1, dilation=dilation, norm=norm, act=act, bias=bias)
        self.conv2 = Conv2DLayer(channel, channel, kernel_size=3, stride=1, padding=1, dilation=dilation, norm=norm, act=None, bias=None)
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
    def __init__(self, in_channels, out_channels, reduction):
        super().__init__()

        self.conv = Conv2DLayer(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.csa0 = ResidualCbamBlock(channel=out_channels, reduction=reduction)
        self.csa1 = ResidualCbamBlock(channel=out_channels, reduction=reduction)
        self.out = nn.Sequential(
            Conv2DLayer(out_channels, out_channels, 1),
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
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Sequential(
            Conv2DLayer(out_channels, out_channels, kernel_size=3, padding=1),
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
        self.se3 = SEResidualBlock(channel=6, se_reduction=2, res_scale=0.1)

        self.se4 = SEResidualBlock(channel=18, se_reduction=6, res_scale=0.1)
        self.se5 = SEResidualBlock(channel=18, se_reduction=6, res_scale=0.1)
        self.se6 = SEResidualBlock(channel=18, se_reduction=6, res_scale=0.1)
        self.se7 = SEResidualBlock(channel=18, se_reduction=6, res_scale=0.1)

        # Output
        self.out_head = Conv2DLayer(in_channels=18,out_channels=3,kernel_size=1,padding=0,stride=1,bias=False)          
        
        self.tanh = nn.Tanh()
        
        

    def forward(self, x):

        lap,xd8,xd4,xd2 = self.Lap(x) # B 12 H W 和 B,3,H,W

        x_se = torch.cat([x, x],dim=1) # B 6 256 256 扩展是因为se要压
        x_se = self.se0(x_se)
        x_se = self.se1(x_se)
        x_se = self.se2(x_se)
        x_se = self.se3(x_se)

        x_se = torch.cat([x_se, lap], dim=1) # B 6+12 256 256
        x_se = self.se4(x_se)
        x_se = self.se5(x_se)
        x_se = self.se6(x_se)
        x_se = self.se7(x_se)

        out = self.out_head(x_se) # (B,3,256,256)
        out = (self.tanh(out)+1)/2
        return out,xd8,xd4,xd2






class MambaBlock2D(nn.Module):
    def __init__(self, dim, num_blocks=1):
        super().__init__()
        
        # 创建多个Mamba块
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(dim),  # 恢复LayerNorm，稳定训练
                Mamba(dim)
            ))

    def forward(self, x_emb):# x_emb (B,L,C) 

        # with torch.amp.autocast('cuda',enabled=True):
        # x_emb = torch.clamp(x_emb, -10.0, 10.0)
        for block in self.blocks:
            res = x_emb
            x_emb = block(x_emb) + res  # 改为0.5，减少信息衰减

        return x_emb # (B,L,C)

# --------------------------
# Swin Transformer
# --------------------------
class ResidualSwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size, shift_size, mlp_ratio, swin_blocks):
        super().__init__()
        self.token_H = input_resolution[0]
        self.window_size = window_size
        self.shift_size = shift_size
        self.swin_blocks = swin_blocks


        self.blocks = nn.ModuleList()
        for _ in range(swin_blocks):
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(dim),  # 恢复LayerNorm，稳定训练
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size,
                    mlp_ratio=mlp_ratio
                )
            ))

    def forward(self, x_emb):# token (B, H/4, W/4, embedim)

        if self.token_H % self.window_size != 0:
            print(f"Warning: Token height {self.token_H} is not divisible by window size {self.window_size}. Padding may be needed.")
        if self.window_size //2 != self.shift_size:
            print(f"Warning: Window size {self.window_size} and shift size {self.shift_size} are not compatible.")

        # x_emb = torch.clamp(x_emb, -10.0, 10.0)
        res = x_emb
        for block in self.blocks:
            x_emb = block(x_emb) + res  # 改为0.5，减少信息衰减


        return x_emb # token (B, H/4, W/4, embedim)





# --------------------------
# Swin Transformer + Mamba Block
# --------------------------
class MambaSwinBlock(nn.Module):
    def __init__(self, in_channels, out_channels, input_resolution, img_size=(256,256), patch_size=4, window_size=7, embed_dim=96, swin_blocks=2, mamba_blocks=2):
        super().__init__()
        self.input_resolution = input_resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.img_size = img_size
        self.swin_blocks = swin_blocks
        self.mamba_blocks = mamba_blocks

        self.norm_act = nn.Sequential(
            nn.GroupNorm(1,in_channels),  
            nn.GELU()             # 添加 PReLU 激活
        )

        self.patch_embed = PatchEmbed(
            img_size=self.img_size,       # 输入图像大小（支持 tuple）
            patch_size=self.patch_size,       # patch 大小（4×4）
            in_chans=self.in_channels,         # 输入通道数（RGB）
            embed_dim=self.embed_dim,        # 输出 token 维度
            flatten=False,
            norm_layer=LayerNorm2d
        )        

        if self.mamba_blocks > 0:
            self.mamba = MambaBlock2D(dim=self.embed_dim,num_blocks=self.mamba_blocks)     

        if self.swin_blocks>0:
            self.swin = ResidualSwinBlock(
                    dim=self.embed_dim,
                    input_resolution=self.input_resolution,
                    num_heads=self.embed_dim // 32,
                    window_size=self.window_size,
                    shift_size=self.window_size//2,
                    mlp_ratio=4.0,
                    swin_blocks = swin_blocks
                ) 
            

        if swin_blocks>0:
            if self.patch_size == 2:
                self.decoder0_swin = nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, 128, 4, stride=2, padding=1),  # (H/2 → H)
                    nn.ReLU(),
                    Conv2DLayer(128, 64, 3, stride=1, padding=1),          # (2C → C)
                    nn.ReLU(),
                    Conv2DLayer(64, out_channels, 3, padding=1)                                # → RGB残差
                )
            if self.patch_size == 4:
                self.decoder1_swin = nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, 128, 4, stride=2, padding=1),  # (H/4 → H/2)
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),          # (H/2 → H)
                    nn.ReLU(),
                    Conv2DLayer(64, out_channels, 3, padding=1)                                # → RGB残差
                )
            if self.patch_size == 8:
                self.decoder2_swin = nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, 128, 4, stride=2, padding=1),  # (H/8 → H/4)
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),          # (H/4 → H/2)
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),          # (H/2 → H)
                    nn.ReLU(),
                    Conv2DLayer(32, out_channels, 3, padding=1)                                # → RGB残差
                )
            if self.patch_size == 16:
                self.decoder3_swin = nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, 128, 4, stride=2, padding=1),  # (H/16 → H/8)
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),          # (H/8 → H/4)
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),          # (H/4 → H/2)
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),          # (H/2 → H)
                    nn.ReLU(),
                    Conv2DLayer(16, out_channels, 3, padding=1)                                # → RGB残差
                )
        if mamba_blocks>0:
            if self.patch_size == 2:
                self.decoder0_mam = nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, 128, 4, stride=2, padding=1),  # (H/2 → H)
                    nn.ReLU(),
                    Conv2DLayer(128, 64, 3, stride=1, padding=1),          # (2C → C)
                    nn.ReLU(),
                    Conv2DLayer(64, out_channels, 3, padding=1)                                # → RGB残差
                )
            if self.patch_size == 4:
                self.decoder1_mam = nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, 128, 4, stride=2, padding=1),  # (H/4 → H/2)
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),          # (H/2 → H)
                    nn.ReLU(),
                    Conv2DLayer(64, out_channels, 3, padding=1)                                # → RGB残差
                )
            if self.patch_size == 8:
                self.decoder2_mam = nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, 128, 4, stride=2, padding=1),  # (H/8 → H/4)
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),          # (H/4 → H/2)
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),          # (H/2 → H)
                    nn.ReLU(),
                    Conv2DLayer(32, out_channels, 3, padding=1)                                # → RGB残差
                )
            if self.patch_size == 16:
                self.decoder3_mam = nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, 128, 4, stride=2, padding=1),  # (H/16 → H/8)
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),          # (H/8 → H/4)
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),          # (H/4 → H/2)
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),          # (H/2 → H)
                    nn.ReLU(),
                    Conv2DLayer(16, out_channels, 3, padding=1)                                # → RGB残差
                )

        # 新增：两分支的可学习缩放
        self.gamma_mam = nn.Parameter(torch.tensor(2.0))
        self.gamma_swin = nn.Parameter(torch.tensor(1.0))
        # 为了在 state.log 里更易观察，包一层 Identity 让监控钩子能记录
        self.mam_gain = nn.Identity()
        self.swin_gain = nn.Identity()       
    
        self.aaf = AAF(in_channels=out_channels, num_inputs=2)  



    def forward(self, x): # in (B,4,H,W)
        if x.shape[2] % (self.patch_size * self.window_size) !=0:
            print(f"Warning: Input height {x.shape[2]} is not divisible by patch size {self.patch_size} and window size {self.window_size}. Padding may be needed.")
        
        if self.img_size[0] != self.patch_size * self.input_resolution[0]:
            print('Warning: wrong size! input_resolution should be imgsize/patchsize')


        # x = self.norm_act(x)
        # x = torch.clamp(x, -10.0, 10.0)
        b,c,h,w = x.shape
        x_emb = self.patch_embed(x) # flatten False x_emb(B,C,H,W)
        B,C,H,W = x_emb.shape
        x_emb = x_emb.permute(0, 2, 3, 1) # be (B,H,W,C)

        # mamba need (B,L,C)
        if self.mamba_blocks>0:
            x_mam = x_emb.reshape(B, H*W, C)
            x_mam = self.mamba(x_mam)                               
            x_mam = x_mam.permute(0,2,1).reshape(B,C,H,W)

            if self.patch_size == 16:
                x_mam = self.decoder3_mam(x_mam) # (B, 3, H, W)
            if self.patch_size == 8:
                x_mam = self.decoder2_mam(x_mam) # (B, 3, H, W)
            if self.patch_size == 4:
                x_mam = self.decoder1_mam(x_mam) # (B, 3, H, W)
            if self.patch_size == 2:
                x_mam = self.decoder0_mam(x_mam) # (B, 3, H, W)      
            x_mam = self.mam_gain(self.gamma_mam * x_mam)           
        else:
            # x_mam = torch.zeros(b,self.out_channels,h,w).to('cuda') 
            x_mam = x_swin 

        # swin need (B,H,W,C)
        if self.swin_blocks>0:

            x_swin = self.swin(x_emb)  # (B, H/4, W/4, embedim)

            x_swin = x_swin.permute(0, 3, 1, 2)

            if self.patch_size == 16:
                x_swin = self.decoder3_swin(x_swin) # (B, 3, H, W)
            if self.patch_size == 8:
                x_swin = self.decoder2_swin(x_swin) # (B, 3, H, W)
            if self.patch_size == 4:
                x_swin = self.decoder1_swin(x_swin) # (B, 3, H, W)
            if self.patch_size == 2:
                x_swin = self.decoder0_swin(x_swin) # (B, 3, H, W)      
            x_swin = self.swin_gain(self.gamma_swin * x_swin)  # 应用缩放（可监控）           
        else:
            # x_swin = torch.zeros(b,self.out_channels,h,w).to('cuda')
            x_swin = x_mam

        # print(b,c,h,w)
        # print("x_mam shape:",x_mam.shape)
        # print("x_swin shape:",x_swin.shape)

        out = self.aaf([x_mam, x_swin])  # 因为块内已经残差连接 不需要总的残差连接了

        return out
    

# --------------------------
# SubNet 多尺度融合模块
# --------------------------
class SubNet(nn.Module):
    def __init__(self, in_dims=(6, 6, 6, 6), swin_num=0, mamba_num=10):
        super().__init__()

        self.m = MambaSwinBlock(in_channels=6, out_channels=6, img_size=(256,256), patch_size=4 , embed_dim=192 , input_resolution=(64 , 64), window_size=8, swin_blocks=swin_num, mamba_blocks=mamba_num)

        # Level 0: 
        self.aaf0 = AAF(in_channels=in_dims[0], num_inputs=2)

        # Level 1: c0和c2融合
        self.aaf1 = AAF(in_channels=in_dims[0], num_inputs=2)

        # Level 2: c1和c3融合
        self.aaf2 = AAF(in_channels=in_dims[0], num_inputs=2)


        
        shortcut_scale_init_value = 0.5
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, in_dims[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, in_dims[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, in_dims[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, in_dims[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None

        self.c0_view = nn.Identity()
        self.c1_view = nn.Identity()
        self.c2_view = nn.Identity()
        self.c3_view = nn.Identity()
    
    def safe_add(self, x, y):
        # 相同的实现
        if x.shape != y.shape:
            y = F.interpolate(y, size=x.shape[2:], mode='bicubic', align_corners=False)
        return x + y

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign() # ​​符号保留​​
            data.abs_().clamp_(value) # 将输入张量 data 的每个元素的绝对值限制在 [value, +∞) 范围内
            data *= sign
        
    def forward(self, x, c0, c1, c2, c3):
        self._clamp_abs(self.alpha0.data, 1e-3) 
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)
        
        c0 = self.safe_add(self.alpha0 * c0, self.m(self.aaf0([x, c1])))
        c1 = self.safe_add(self.alpha1 * c1, self.m(self.aaf1([c0, c2])))
        c2 = self.safe_add(self.alpha2 * c2, self.m(self.aaf2([c1, c3])))
        c3 = self.safe_add(self.alpha3 * c3, self.m(c2))

        c0 = self.c0_view(c0)
        c1 = self.c1_view(c1)
        c2 = self.c2_view(c2)
        c3 = self.c3_view(c3)

        return c0, c1, c2, c3


# --------------------------
# 解码器
# --------------------------
class Decoder1(nn.Module):
    def __init__(self,in_channels, num_inputs):
        super().__init__()

        self.aaf = AAF(in_channels=in_channels,num_inputs=num_inputs)

    def forward(self, c0, c1, c2, c3):

        x = self.aaf([c0,c1,c2,c3])

        return x
    
class Decoder2(nn.Module):
    def __init__(self, in_dims=(6, 6, 6, 6), num_layers=(4, 4, 4, 4)):
        super().__init__()
        
        self.m = MambaSwinBlock(in_channels=in_dims[0], out_channels=in_dims[0], img_size=(256,256), patch_size=4 , embed_dim=192 , input_resolution=(64 , 64), window_size=8, swin_blocks=0, mamba_blocks=num_layers[0])
        self.norm0 = nn.BatchNorm2d(in_dims[0])
        self.norm1 = nn.BatchNorm2d(in_dims[0])
        self.norm2 = nn.BatchNorm2d(in_dims[0])

        self.xc2_view = nn.Identity()
        self.xc1_view = nn.Identity()
        self.xc0_view = nn.Identity()
    def forward(self, x_in, c0, c1, c2, c3):

        x = self.m(c3)
        x = self.norm0(x)
        x = self.m(self.xc2_view(x*c2))
        x = self.norm1(x)
        x = self.m(self.xc1_view(x*c1))
        x = self.norm2(x)
        x = self.m(self.xc0_view(x*c0))

        return x



# --------------------------
# 主模型
# --------------------------
class MTRRNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.norm_first = nn.BatchNorm2d(3)

        self.rdm = RDM()  # 提取高光区域图

        # 编码器三阶段：Swin+Mamba 堆叠结构
        # input_resolution要是window_size的倍数
        # input H/W 推荐是 patch_size 和 window_size 的公倍数
        # 输入256*256
        self.encoder0 = MambaSwinBlock(in_channels=3, out_channels=3, img_size=(256,256), patch_size=4 , embed_dim=96 , input_resolution=(64 , 64), window_size=8, swin_blocks=4, mamba_blocks=10) # 最细节的
        # 输入128*128
        self.encoder1 = MambaSwinBlock(in_channels=3, out_channels=3, img_size=(128,128), patch_size=8 , embed_dim=192, input_resolution=(16 , 16), window_size=8, swin_blocks=4, mamba_blocks=10)
        # 输入64*64
        self.encoder2 = MambaSwinBlock(in_channels=3, out_channels=3, img_size=(64 , 64), patch_size=4 , embed_dim=96 , input_resolution=(16 , 16), window_size=8, swin_blocks=4, mamba_blocks=10)
        # 输入32*32
        self.encoder3 = MambaSwinBlock(in_channels=3, out_channels=3, img_size=(32 , 32), patch_size=2 , embed_dim=96 , input_resolution=(16 , 16), window_size=4, swin_blocks=4, mamba_blocks=10)

        # 特征通道适配
        self.c0_adapter = nn.Sequential(
            Conv2DLayer(3, 6, 1, norm=nn.BatchNorm2d),
            nn.GELU()
        )
        self.c1_adapter = nn.Sequential(
            Conv2DLayer(3, 6, 1, norm=nn.BatchNorm2d),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 上采样至 256×256
        )
        self.c2_adapter = nn.Sequential(
            Conv2DLayer(3, 6, 1, norm=nn.BatchNorm2d),
            nn.GELU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)  # 上采样至 256×256
        )
        self.c3_adapter = nn.Sequential(
            Conv2DLayer(3, 6, 1, norm=nn.BatchNorm2d),
            nn.GELU(),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)  # 上采样至 256×256
        )


        # 三层 SubNet，不共享参数
        self.subnet0 = SubNet(in_dims=(6, 6, 6, 6), swin_num=0, mamba_num=10)
        self.subnet1 = SubNet(in_dims=(6, 6, 6, 6), swin_num=0, mamba_num=10)
        # self.subnet2 = SubNet(in_dims=(6, 6, 6, 6), num_layers=(2, 2, 2, 2))
        # self.subnet3 = SubNet(in_dims=(6, 6, 6, 6), num_layers=(2, 2, 2, 2))

        # 解码器
        # self.decoder1 = Decoder1(6,4)
        self.decoder2 = Decoder2(in_dims=(6, 6, 6, 6), num_layers=(2, 2, 2, 2))

        self.down8 = nn.Identity()
 

    def run_subnet0(self, *inputs): return self.subnet0(*inputs)
    def run_subnet1(self, *inputs): return self.subnet1(*inputs)
    # def run_subnet2(self, *inputs): return self.subnet2(*inputs)
    # def run_subnet3(self, *inputs): return self.subnet3(*inputs)

    def forward(self, x_in):
        # x_in = self.norm_first(x_in)
        rmap,x_down8,x_down4,x_down2 = self.rdm(x_in)  # 提取反光区域图 都是c=3
        x_down1 = x_in

        # rmap = 1-rmap  # 反光区域图取反才是抑制反光

        rmapd2 = F.interpolate(rmap, scale_factor=0.5, mode='bicubic')
        rmapd4 = F.interpolate(rmap, scale_factor=0.25, mode='bicubic')
        rmapd8 = F.interpolate(rmap, scale_factor=0.125, mode='bicubic')# 下采样到 1/8

        # x_down1 = torch.cat([x_down1, rmap], dim=1)  # B, 4, 128, 128
        # x_down2 = torch.cat([x_down2, rmapd2], dim=1)
        # x_down4 = torch.cat([x_down4, rmapd4], dim=1)
        # x_down8 = torch.cat([x_down8, rmapd8], dim=1)  # B, 4, 32, 32
        # x_down1 = x_down1 * rmap    # B, 4, 128, 128
        # x_down2 = x_down2 * rmapd2
        # x_down4 = x_down4 * rmapd4
        # x_down8 = x_down8 * rmapd8  # B, 4, 32, 32

        x_down8 = self.down8(x_down8) # 观察

        x_down1 = self.encoder0(x_down1)
        x_down2 = self.encoder1(x_down2)
        x_down4 = self.encoder2(x_down4)
        x_down8 = self.encoder3(x_down8)
        

        # x_down1 = checkpoint.checkpoint(self.encoder0, x_down1)
        # x_down2 = checkpoint.checkpoint(self.encoder1, x_down2)
        # x_down4 = checkpoint.checkpoint(self.encoder2, x_down4)
        # x_down8 = checkpoint.checkpoint(self.encoder3, x_down8)

        # 通道和分辨率转换
        c0 = self.c0_adapter(x_down1) # 全变成 c=6
        c1 = self.c1_adapter(x_down2)
        c2 = self.c2_adapter(x_down4)
        c3 = self.c3_adapter(x_down8)

        # # 四层子网络增强
        x = torch.cat([x_in,x_in],dim=1)
        # c0, c1, c2, c3 = self.subnet0(x, c0, c1, c2, c3)
        # c0, c1, c2, c3 = self.subnet1(x, c0, c1, c2, c3)
        # c0, c1, c2, c3 = self.subnet2(x, c0, c1, c2, c3)
        # c0, c1, c2, c3 = self.subnet3(x, c0, c1, c2, c3) # 都是6通道的

 
        c0, c1, c2, c3 = checkpoint.checkpoint(self.run_subnet0, x, c0, c1, c2, c3)
        c0, c1, c2, c3 = checkpoint.checkpoint(self.run_subnet1, x, c0, c1, c2, c3)
        # c0, c1, c2, c3 = checkpoint.checkpoint(self.run_subnet2, x, c0, c1, c2, c3)
        # c0, c1, c2, c3 = checkpoint.checkpoint(self.run_subnet3, x, c0, c1, c2, c3)    



        # 解码重建残差图
        # 修复：使用x_in初始化两个通道，避免fake_R无法学习
        out = torch.cat([x_in, x_in * 0.1],dim=1) + self.decoder2(x, c0, c1, c2, c3)
        # out = torch.cat([x_in,x_in],dim=1) + (c0)
        # out = self.decoder2(rmap2, c0, c1, c2, c3)
        # out = self.decoder1(c0, c1, c2, c3)

        return rmap, out

 

class MTRREngine(nn.Module):

    def __init__(self, opts, device):
        super(MTRREngine, self).__init__()
        self.device = device 
        self.opts  = opts
        self.visual_names = ['fake_T', 'fake_R', 'c_map', 'I', 'Ic', 'T', 'R']
        self.netG_T = MTRRNet().to(device)  
        self.netG_T.apply(self.init_weights)
        self.net_c = PretrainedConvNext_e2e("convnext_small_in22k").cuda()
        # print(torch.load('./pretrained/cls_model.pth', map_location=str(self.device)).keys())
        self.net_c.load_state_dict(torch.load('./cls/cls_models/clsbest.pth', map_location=str(self.device)))
        self.net_c.eval()  # 预训练模型不需要训练        



    def load_checkpoint(self, optimizer):
        if self.opts.model_path is not None:
            model_path = self.opts.model_path
            print('Load the model from %s' % model_path)
            model_state = torch.load(model_path, map_location=str(self.device))
            
            self.netG_T.load_state_dict({k.replace('netG_T.', ''): v for k, v in model_state['netG_T'].items()})

            if 'optimizer_state_dict' in model_state:
                try:
                    optimizer.load_state_dict(model_state['optimizer_state_dict'])
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = model_state.get('lr', param_group['lr'])
                except ValueError as e:
                    print(f"Warning: Could not load optimizer state due to: {e}")
                    print("Continuing with fresh optimizer state")
                    # 只设置学习率，不加载整个state
                    if 'lr' in model_state:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = model_state['lr']

            epoch = model_state.get('epoch', None)
            print('Loaded model at epoch %d' % (epoch+1) if epoch is not None else 'Loaded model without epoch info')
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
        with torch.no_grad():
            self.Ic = self.net_c(self.I)
        self.c_map, self.out = self.netG_T(self.Ic) 
        self.fake_T, self.fake_R = self.out[:,0:3,:,:],self.out[:,3:6,:,:]


        
 
    def monitor_layer_stats(self):
        """仅监控模型的一级子模块（不深入嵌套结构）"""
        hooks = []
        model = self.netG_T

        # 修正钩子函数参数（正确接收module, input, output）
        def _hook_fn(module, input, output, layer_name):
            if isinstance(output, torch.Tensor):
                mean = output.mean().item()
                std = output.std().item()

                is_nan = math.isnan(mean) or math.isnan(std)
                if is_nan or self.opts.always_print:
                    msg = f"{layer_name:<50} | Mean: {mean:>15.6f} | Std: {std:>15.6f} | Shape: {tuple(output.shape)}"
                    # print(msg)
                    with open('./debug/state.log', 'a') as f:
                        f.write(msg + '\n')# 修正钩子函数参数（正确接收module, input, output）
      

        # 遍历所有子模块并注册钩子
        for name, module in model.named_modules():
            if not isinstance(module, nn.ModuleList):  # 过滤容器类（如Sequential）
                hook = module.register_forward_hook(
                    lambda m, inp, out, name=name: _hook_fn(m, inp, out, name)
                )
                hooks.append(hook)   
        

    def monitor_layer_grad(self):
        with open('./debug/grad.log', 'a') as f:
            for name, param in self.netG_T.named_parameters():

                if param.grad is not None:
                    is_nan = math.isnan(param.grad.mean().item()) or math.isnan(param.grad.std().item())
                    if is_nan or self.opts.always_print:
                        if param.grad is not None:
                            msg = (
                                f"Param: {name:<50} | "
                                f"Grad Mean: {param.grad.mean().item():.15f} | "
                                f"Grad Std: {param.grad.std().item():.15f}"
                            )
                        else:
                            msg = f"Param: {name:<50} | Grad is None"  # 梯度未回传  
                        # print(msg)
                        f.write(msg + '\n')

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
