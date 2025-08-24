# Token-based Modules for MTRRNet
# 实现Token-only多尺度编码→融合→统一解码架构
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from timm.models.vision_transformer import PatchEmbed
from timm.models.swin_transformer import SwinTransformerBlock
from timm.layers import LayerNorm2d
import math


class FrequencySplit(nn.Module):
    """频带分离：输入图像分离为低频（模糊）和高频部分"""
    def __init__(self, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        # 使用可学习的高斯核进行低频提取
        self.register_parameter('blur_weight', nn.Parameter(torch.ones(1)))
        
    def forward(self, x):
        # 创建高斯模糊核
        kernel = self._create_gaussian_kernel(self.kernel_size, x.device)
        kernel = kernel.expand(x.size(1), 1, self.kernel_size, self.kernel_size)
        
        # 低频：高斯模糊
        padding = self.kernel_size // 2
        low_freq = F.conv2d(x, kernel, padding=padding, groups=x.size(1))
        low_freq = low_freq * self.blur_weight
        
        # 高频：原图减去低频
        high_freq = x - low_freq
        
        return low_freq, high_freq
    
    def _create_gaussian_kernel(self, kernel_size, device):
        """创建高斯核"""
        coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2
        grid = coords.unsqueeze(0).expand(kernel_size, -1)
        kernel = torch.exp(-(grid ** 2 + grid.T ** 2) / (2 * (kernel_size / 6) ** 2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)


class TokenPatchEmbed(nn.Module):
    """将图像patches转换为tokens"""
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        # x: (B, C, H, W) -> (B, embed_dim, H//patch_size, W//patch_size)
        x = self.proj(x)
        # -> (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x  # (B, num_patches, embed_dim)


class MambaTokenBlock(nn.Module):
    """Mamba处理token序列，采用x = x + f(LN(x))残差连接"""
    def __init__(self, dim, num_blocks=1):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(dim),
                Mamba(dim)
            ))
    
    def forward(self, x):
        # x: (B, N, C)
        for block in self.blocks:
            x = x + block(x)  # 逐层残差: x = x + f(LN(x))
        return x


class SwinTokenBlock(nn.Module):
    """Swin Transformer处理token，采用x = x + f(LN(x))残差连接"""
    def __init__(self, dim, input_resolution, num_heads, window_size, num_blocks=1):
        super().__init__()
        self.input_resolution = input_resolution
        self.blocks = nn.ModuleList()
        
        for i in range(num_blocks):
            shift_size = 0 if i % 2 == 0 else window_size // 2
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(dim),
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size,
                    mlp_ratio=4.0
                )
            ))
    
    def forward(self, x):
        # x: (B, N, C) 需要reshape为(B, H, W, C)用于Swin
        B, N, C = x.shape
        H, W = self.input_resolution
        
        # 转换为Swin格式
        x = x.view(B, H, W, C)
        
        for block in self.blocks:
            residual = x
            x = block(x) + residual  # 逐层残差: x = x + f(LN(x))
        
        # 转换回token格式
        x = x.view(B, N, C)
        return x


class TokenStage(nn.Module):
    """单个尺度的Token处理阶段：频带分离→分别编码→融合"""
    def __init__(self, img_size, patch_size, in_chans, embed_dim, 
                 mamba_blocks=2, swin_blocks=2, window_size=8):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        
        # 频带分离
        self.freq_split = FrequencySplit(kernel_size=5)
        
        # 低频和高频的patch embedding
        self.low_embed = TokenPatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.high_embed = TokenPatchEmbed(img_size, patch_size, in_chans, embed_dim)
        
        # Mamba处理低频，Swin处理高频
        if mamba_blocks > 0:
            self.mamba_processor = MambaTokenBlock(embed_dim, mamba_blocks)
        else:
            self.mamba_processor = None
            
        if swin_blocks > 0:
            input_resolution = (self.grid_size, self.grid_size)
            num_heads = max(1, embed_dim // 32)
            self.swin_processor = SwinTokenBlock(
                embed_dim, input_resolution, num_heads, window_size, swin_blocks)
        else:
            self.swin_processor = None
        
        # 融合模块：加权平均或concat+线性
        if self.mamba_processor is not None and self.swin_processor is not None:
            self.fusion_type = 'weighted'  # 或 'concat'
            if self.fusion_type == 'weighted':
                self.fusion_weight = nn.Parameter(torch.tensor(0.5))
            else:
                self.fusion_proj = nn.Linear(embed_dim * 2, embed_dim)
        
        # 中间监督head（可选）
        self.aux_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 3)  # 输出3通道用于aux_T
        )
        
    def forward(self, x):
        # 频带分离
        low_freq, high_freq = self.freq_split(x)  # (B, C, H, W)
        
        # 转换为tokens
        low_tokens = self.low_embed(low_freq)   # (B, N, C)
        high_tokens = self.high_embed(high_freq)  # (B, N, C)
        
        # 分别处理
        if self.mamba_processor is not None:
            low_tokens = self.mamba_processor(low_tokens)
        else:
            low_tokens = torch.zeros_like(low_tokens)
            
        if self.swin_processor is not None:
            high_tokens = self.swin_processor(high_tokens)
        else:
            high_tokens = torch.zeros_like(high_tokens)
        
        # 融合
        if self.mamba_processor is not None and self.swin_processor is not None:
            if self.fusion_type == 'weighted':
                w = torch.sigmoid(self.fusion_weight)
                fused_tokens = w * low_tokens + (1 - w) * high_tokens
            else:
                concat_tokens = torch.cat([low_tokens, high_tokens], dim=-1)
                fused_tokens = self.fusion_proj(concat_tokens)
        elif self.mamba_processor is not None:
            fused_tokens = low_tokens
        else:
            fused_tokens = high_tokens
        
        # 中间监督（用于可视化）
        aux_pred = None
        if hasattr(self, 'aux_head'):
            # 从tokens重建为图像用于aux supervision
            B, N, C = fused_tokens.shape
            H = W = self.grid_size
            # 平均池化tokens并重建
            aux_tokens = fused_tokens.mean(dim=1)  # (B, C)
            aux_pred = self.aux_head(aux_tokens)   # (B, 3)
            aux_pred = aux_pred.view(B, 3, 1, 1).expand(-1, -1, self.img_size, self.img_size)
        
        return fused_tokens, aux_pred


class MultiScaleTokenEncoder(nn.Module):
    """多尺度Token编码器：处理4个尺度得到token表示"""
    def __init__(self, embed_dims=[192, 192, 96, 96], 
                 mamba_blocks=[2, 2, 2, 2], swin_blocks=[2, 2, 2, 2]):
        super().__init__()
        self.scales = [256, 128, 64, 32]  # 对应encoder0~3的分辨率
        self.patch_sizes = [8, 4, 4, 2]  # 每个尺度的patch size
        
        self.stages = nn.ModuleList()
        for i, (scale, patch_size, embed_dim, mb, sb) in enumerate(
            zip(self.scales, self.patch_sizes, embed_dims, mamba_blocks, swin_blocks)):
            
            stage = TokenStage(
                img_size=scale,
                patch_size=patch_size,
                in_chans=3,
                embed_dim=embed_dim,
                mamba_blocks=mb,
                swin_blocks=sb,
                window_size=8
            )
            self.stages.append(stage)
    
    def forward(self, x_in):
        # x_in: (B, 3, 256, 256)
        tokens_list = []
        aux_preds = {}
        
        for i, stage in enumerate(self.stages):
            # 下采样到对应尺度
            scale = self.scales[i]
            if scale != 256:
                x_scale = F.interpolate(x_in, size=(scale, scale), mode='bilinear', align_corners=False)
            else:
                x_scale = x_in
            
            # 获得该尺度的token表示
            tokens, aux_pred = stage(x_scale)
            tokens_list.append(tokens)
            
            if aux_pred is not None:
                aux_preds[f'aux_s{i}'] = aux_pred
        
        return tokens_list, aux_preds


class TokenSubNet(nn.Module):
    """Token融合/细化模块：多尺度token交互融合"""
    def __init__(self, ref_resolution=64, embed_dim=192, num_blocks=3):
        super().__init__()
        self.ref_resolution = ref_resolution
        self.embed_dim = embed_dim
        
        # 统一到参考分辨率的投影层
        self.projectors = nn.ModuleList([
            nn.Linear(192, embed_dim),  # s0: 256->ref
            nn.Linear(192, embed_dim),  # s1: 128->ref  
            nn.Linear(96, embed_dim),   # s2: 64->ref
            nn.Linear(96, embed_dim),   # s3: 32->ref
        ])
        
        # 1x1 conv for channel fusion after concat
        self.fusion_conv = nn.Conv2d(embed_dim * 4, embed_dim, 1)
        
        # 融合后的细化处理
        self.refinement_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.refinement_blocks.append(nn.Sequential(
                nn.LayerNorm(embed_dim),
                Mamba(embed_dim)
            ))
    
    def forward(self, tokens_list):
        # tokens_list: [t0, t1, t2, t3] 每个是 (B, N_i, C_i)
        B = tokens_list[0].shape[0]
        ref_H = ref_W = self.ref_resolution
        
        # 将所有tokens对齐到参考分辨率
        aligned_tokens = []
        for i, (tokens, proj) in enumerate(zip(tokens_list, self.projectors)):
            # 投影维度
            tokens = proj(tokens)  # (B, N_i, embed_dim)
            
            # 重建为spatial形式
            N_i = tokens.shape[1]
            H_i = W_i = int(math.sqrt(N_i))
            tokens_spatial = tokens.view(B, H_i, W_i, self.embed_dim).permute(0, 3, 1, 2)
            
            # 插值到参考分辨率
            if H_i != ref_H or W_i != ref_W:
                tokens_spatial = F.interpolate(
                    tokens_spatial, size=(ref_H, ref_W), mode='bilinear', align_corners=False)
            
            aligned_tokens.append(tokens_spatial)
        
        # 在通道维度concat并融合
        fused_spatial = torch.cat(aligned_tokens, dim=1)  # (B, embed_dim*4, ref_H, ref_W)
        fused_spatial = self.fusion_conv(fused_spatial)   # (B, embed_dim, ref_H, ref_W)
        
        # 转回token格式进行细化
        fused_tokens = fused_spatial.flatten(2).transpose(1, 2)  # (B, ref_H*ref_W, embed_dim)
        
        # 细化处理
        for block in self.refinement_blocks:
            fused_tokens = fused_tokens + block(fused_tokens)  # 逐层残差
        
        return fused_tokens  # (B, ref_H*ref_W, embed_dim)


class UnifiedTokenDecoder(nn.Module):
    """统一Token解码器：从token一次性解码为6通道(T,R)"""
    def __init__(self, token_dim=192, ref_resolution=64, base_scale_init=0.3):
        super().__init__()
        self.ref_resolution = ref_resolution
        self.token_dim = token_dim
        
        # Base缩放因子
        self.base_scale = nn.Parameter(torch.tensor(base_scale_init))
        
        # Token到feature map的转换
        self.token_to_feature = nn.Sequential(
            nn.Linear(token_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256)
        )
        
        # 上采样和卷积解码层
        self.decoder = nn.Sequential(
            # 64x64 -> 128x128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            
            # 128x128 -> 256x256  
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            
            # 最终输出层
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 6, kernel_size=1)  # 6通道输出 (T, R)
        )
        
    def forward(self, tokens, x_in):
        # tokens: (B, ref_H*ref_W, token_dim)
        # x_in: (B, 3, 256, 256) 原始输入
        B = tokens.shape[0]
        ref_H = ref_W = self.ref_resolution
        
        # 转换tokens到feature map
        feature_tokens = self.token_to_feature(tokens)  # (B, ref_H*ref_W, 256)
        feature_map = feature_tokens.view(B, ref_H, ref_W, 256).permute(0, 3, 1, 2)
        
        # 解码
        delta = self.decoder(feature_map)  # (B, 6, 256, 256)
        
        # Base residual: 输入图像的residual base
        base_input = x_in.repeat(1, 2, 1, 1)  # (B, 6, 256, 256) 复制为T,R base
        base = self.base_scale * base_input
        
        # 最终输出 = base + delta
        output = base + delta
        
        return output  # (B, 6, 256, 256)