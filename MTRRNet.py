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



class MTRRNet(nn.Module):
    """Token-only多尺度架构：
    多尺度编码 → Token融合 → 统一解码
    频带分工：低频→Mamba，高频→Swin
    """
    def __init__(self, use_legacy=False):
        super().__init__()
        self.use_legacy = use_legacy
        
        if use_legacy:
            # 使用旧实现的组件
            self._init_legacy()
        else:
            # 新Token-only实现
            self._init_token_only()
        
    def _init_token_only(self):
        """初始化Token-only版本的组件"""
        # 延迟导入token模块以避免循环导入和依赖问题
        from token_modules import (
            MultiScaleTokenEncoder, TokenSubNet, UnifiedTokenDecoder
        )
        
        # RDM保持不变，用于生成rmap
        self.rdm = RDM()
        
        # 多尺度Token编码器
        self.token_encoder = MultiScaleTokenEncoder(
            embed_dims=[192, 192, 96, 96],    # 对应原encoder0~3的embed_dim  
            mamba_blocks=[10, 10, 10, 10],    # Mamba处理低频
            swin_blocks=[4, 4, 4, 4]          # Swin处理高频
        )
        
        # Token SubNet：多尺度token融合
        self.token_subnet = TokenSubNet(
            ref_resolution=64,     # 统一到64x64分辨率网格
            embed_dim=192,         # 融合后的token维度
            num_blocks=3           # 融合细化的block数
        )
        
        # 统一Token解码器
        self.token_decoder = UnifiedTokenDecoder(
            token_dim=192,         # 输入token维度
            ref_resolution=64,     # token网格分辨率
            base_scale_init=0.3    # base缩放因子初始值
        )
        
        # 用于存储中间监督结果（可视化）
        self.intermediates = {}
        
        # 监控钩子用的debug张量
        self.debug_token_stats = {}
        
    def forward(self, x_in):
        """
        输入: x_in (B, 3, 256, 256)
        输出: (rmap, out) 其中out为6通道(T,R)
        """
        if self.use_legacy:
            return self._forward_legacy(x_in)
        else:
            return self._forward_token_only(x_in)
    
    
    def _forward_token_only(self, x_in):
        """Token-only版本的前向传播"""
        B = x_in.shape[0]
        
        # 1. RDM提取反光先验（保持与原架构兼容）
        rmap, _, _, _ = self.rdm(x_in)  # rmap: (B, 3, 256, 256)
        
        # 2. 多尺度Token编码
        tokens_list, aux_preds = self.token_encoder(x_in)
        # tokens_list: [t0, t1, t2, t3] 每个(B, N_i, C_i)
        # aux_preds: {'aux_s0': pred, ...} 中间监督预测
        
        # 缓存中间监督结果
        self.intermediates.update(aux_preds)
        
        # 缓存token统计用于监控
        for i, tokens in enumerate(tokens_list):
            self.debug_token_stats[f'tokens_s{i}_mean'] = tokens.mean().detach()
            self.debug_token_stats[f'tokens_s{i}_std'] = tokens.std().detach()
        
        # 3. Token SubNet融合
        fused_tokens = self.token_subnet(tokens_list)  # (B, ref_H*ref_W, embed_dim)
        
        # 缓存融合后token统计
        self.debug_token_stats['fused_tokens_mean'] = fused_tokens.mean().detach()
        self.debug_token_stats['fused_tokens_std'] = fused_tokens.std().detach()
        
        # 4. 统一解码：token → 6通道(T,R)
        out = self.token_decoder(fused_tokens, x_in)  # (B, 6, 256, 256)
        
        # 缓存解码输出统计
        self.debug_token_stats['output_mean'] = out.mean().detach()
        self.debug_token_stats['output_std'] = out.std().detach()
        
        return rmap, out


    def get_intermediates(self):
        """获取中间监督结果用于可视化（仅Token-only模式）"""
        return self.intermediates if not self.use_legacy else {}
    
    def get_debug_stats(self):
        """获取debug统计信息（仅Token-only模式）"""
        return self.debug_token_stats if not self.use_legacy else {}


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
        # 暂停net_c：直接使用原始输入self.I（不破坏接口）
        # with torch.no_grad():
        #     self.Ic = self.net_c(self.I)
        self.Ic = self.I  # 直接设置为原始输入
        
        # 使用原始输入调用token-only模型
        # self.c_map, self.out = self.netG_T(self.I) 
        self.c_map, self.out = self.netG_T(self.I)  # 改为使用self.I而非self.Ic
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
        
        # 额外监控token统计信息
        self.monitor_token_stats()

    def monitor_token_stats(self):
        """监控token阶段的统计信息"""
        if hasattr(self.netG_T, 'get_debug_stats'):
            token_stats = self.netG_T.get_debug_stats()
            with open('./debug/token_stats.log', 'a') as f:
                for name, value in token_stats.items():
                    if torch.is_tensor(value):
                        f.write(f"Token {name:<100} | Value: {value.item():>15.6f}\n")
                    else:
                        f.write(f"Token {name:<100} | Value: {value:>15.6f}\n")
        

    def monitor_layer_grad(self):
        with open('./debug/grad.log', 'a') as f:
            for name, param in self.netG_T.named_parameters():

                if param.grad is not None:
                    is_nan = math.isnan(param.grad.mean().item()) or math.isnan(param.grad.std().item())
                    if is_nan or self.opts.always_print:
                        if param.grad is not None:
                            msg = (
                                f"Param: {name:<100} | "
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
