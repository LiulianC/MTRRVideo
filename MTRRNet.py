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



# ================== 0. 通用 Token Swin Transformer 堆叠 ==================

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

        for block in self.blocks:
            res = x_emb
            x_emb = block(x_emb) + res  
        return x_emb # token (B, H/4, W/4, embedim)




# ================== 1. 通用 Token Mamba 堆叠 ==================
class TokenMambaStack(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                Mamba(dim)
            ) for _ in range(depth)
        ])
    def forward(self, x):  # x: (B,L,dim)
        for blk in self.blocks:
            residual = x
            x = blk(x) + residual
        return x

# ================== 2. 单分支编码（PatchEmbed + 并联 Mamba/Swin + AAF） ==================
class SingleScaleEncoderBranch(nn.Module):
    """
    patch_size / embed_dim / swin_blocks / mamba_blocks 可调
    输出 tokens: (B, L=H*W, embed_dim)
    """
    def __init__(self, in_ch, patch_size, embed_dim,
                 img_size=256, input_resolution=None,
                 swin_blocks=4, mamba_blocks=10,
                 window_size=8):
        super().__init__()
        assert input_resolution is not None, "input_resolution 必须提供 (H,W)"
        self.embed_dim = embed_dim
        self.grid_h, self.grid_w = input_resolution
        # 使用 timm 的 PatchEmbed
        self.patch = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_ch,
            embed_dim=embed_dim,
            flatten=False,
            norm_layer=LayerNorm2d
        )
        # 并联子结构
        self.use_mamba = mamba_blocks > 0
        self.use_swin  = swin_blocks  > 0
        if self.use_mamba:
            self.mamba_stack = TokenMambaStack(embed_dim, mamba_blocks)
        if self.use_swin:
            self.swin_stack = ResidualSwinBlock(
                dim=embed_dim,
                input_resolution=input_resolution,
                num_heads=max(1, embed_dim // 32),
                window_size=window_size,
                shift_size=window_size // 2,
                mlp_ratio=4.0,
                swin_blocks=swin_blocks
            )
        self.aaf = AAF(in_channels=embed_dim, num_inputs=2) if (self.use_mamba and self.use_swin) else None

    def forward(self, x):  # x: (B,3,256,256)
        feat = self.patch(x)            # (B, embed_dim, H, W)
        B,C,H,W = feat.shape
        tokens = feat.permute(0,2,3,1).reshape(B, H*W, C)  # (B,L,C)

        if self.use_mamba:
            tm = self.mamba_stack(tokens)  # (B,L,C)
        if self.use_swin:
            ts = self.swin_stack(feat.permute(0,2,3,1))  # (B,H,W,C)
            ts = ts.view(B, H*W, C)
        if self.aaf is None:
            fused = tm if self.use_mamba else ts
        else:
            fm = tm.view(B,H,W,C).permute(0,3,1,2)   # (B,C,H,W)
            fs = ts.view(B,H,W,C).permute(0,3,1,2)
            fout = self.aaf([fm, fs])               # (B,C,H,W)
            fused = fout.flatten(2).transpose(1,2)  # (B,L,C)
        return fused  # (B,L,C)

# ================== 3. 多尺度编码器（3 分支） ==================
class MultiBranchEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # c0: 64x64, embed 96
        self.branch_c0 = SingleScaleEncoderBranch(
            in_ch=3, patch_size=4, embed_dim=96,
            img_size=256, input_resolution=(64,64),
            swin_blocks=4, mamba_blocks=10, window_size=8
        )
        # c1: 32x32, embed 192
        self.branch_c1 = SingleScaleEncoderBranch(
            in_ch=3, patch_size=8, embed_dim=192,
            img_size=256, input_resolution=(32,32),
            swin_blocks=4, mamba_blocks=10, window_size=8
        )
        # c2: 16x16, embed 768
        self.branch_c2 = SingleScaleEncoderBranch(
            in_ch=3, patch_size=16, embed_dim=768,
            img_size=256, input_resolution=(16,16),
            swin_blocks=4, mamba_blocks=10, window_size=2
        )

    def forward(self, x):
        c0 = self.branch_c0(x)  # (B,4096,96)
        c1 = self.branch_c1(x)  # (B,1024,192)
        c2 = self.branch_c2(x)  # (B, 256,768)
        return c0, c1, c2

# ================== 4. 跨尺度适配工具 ==================
class TokenAdapter(nn.Module):
    """提供 上采样 / 下采样 + 线性投影 功能 (H,W 已知)"""
    def __init__(self, in_dim, out_dim, in_hw, out_hw):
        super().__init__()
        self.in_h, self.in_w = in_hw
        self.out_h, self.out_w = out_hw
        self.proj_in = nn.Identity()  # 可按需要调整投影顺序
        self.proj_out = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, tokens):
        # tokens: (B, L_in, C_in) 其中 L_in = in_h * in_w
        B, L, C = tokens.shape
        x = tokens.view(B, self.in_h, self.in_w, C).permute(0,3,1,2)  # (B,C,H,W)
        if self.in_h != self.out_h or self.in_w != self.out_w:
            x = F.interpolate(x, size=(self.out_h, self.out_w), mode='bilinear', align_corners=False)
        x = x.permute(0,2,3,1).reshape(B, self.out_h*self.out_w, C)
        x = self.proj_out(x)  # (B, L_out, out_dim)
        return x

# ================== 5. 融合阶段（两次迭代显式写出） ==================
class TwoIterMultiScaleFusion(nn.Module):
    """
    每个尺度各自有 fusion_mamba 堆叠 (dim = 该尺度 embed_dim)
    迭代 2 次，显式展开，不用循环。
    仅 c1_next 使用 AAF 融合来自 (c0, c2) 适配后的特征。
    """
    def __init__(self):
        super().__init__()
        # 第一尺度的 token 尺寸
        self.c0_hw = (64,64); self.c0_dim = 96
        self.c1_hw = (32,32); self.c1_dim = 192
        self.c2_hw = (16,16); self.c2_dim = 768

        fusion_depth = 5
        # 每尺度自己的融合 Mamba
        self.mamba_c0 = TokenMambaStack(self.c0_dim, fusion_depth)
        self.mamba_c1 = TokenMambaStack(self.c1_dim, fusion_depth)
        self.mamba_c2 = TokenMambaStack(self.c2_dim, fusion_depth)

        self.mamba_c0_2 = TokenMambaStack(self.c0_dim, fusion_depth)
        self.mamba_c1_2 = TokenMambaStack(self.c1_dim, fusion_depth)
        self.mamba_c2_2 = TokenMambaStack(self.c2_dim, fusion_depth)

        # 适配器（c1→c2, c0→c1, c2→c1, c1→c0） 两次迭代共用相同结构（参数共享）
        self.adapt_c1_to_c2 = TokenAdapter(in_dim=self.c1_dim, out_dim=self.c2_dim,
                                           in_hw=self.c1_hw, out_hw=self.c2_hw)
        self.adapt_c0_to_c1 = TokenAdapter(in_dim=self.c0_dim, out_dim=self.c1_dim,
                                           in_hw=self.c0_hw, out_hw=self.c1_hw)
        self.adapt_c2_to_c1 = TokenAdapter(in_dim=self.c2_dim, out_dim=self.c1_dim,
                                           in_hw=self.c2_hw, out_hw=self.c1_hw)
        self.adapt_c1_to_c0 = TokenAdapter(in_dim=self.c1_dim, out_dim=self.c0_dim,
                                           in_hw=self.c1_hw, out_hw=self.c0_hw)

        # 只在 c1 融合时需要 AAF (输入通道 = c1_dim, num_inputs=2)
        self.aaf_c1 = AAF(in_channels=self.c1_dim, num_inputs=2)

    def _fusion_step(self, c0, c1, c2,
                     mamba_c0, mamba_c1, mamba_c2):
        """
        单步：
          c2_next = c2 + M_c2( adapt(c1) )
          c1_next = c1 + M_c1( AAF( adapt(c0), adapt(c2) ) )
          c0_next = c0 + M_c0( adapt(c1) )
        """
        # c2 更新
        c1_down_to_c2 = self.adapt_c1_to_c2(c1)           # (B,256,768)
        c2_delta = mamba_c2(c1_down_to_c2)                # (B,256,768)
        c2_next = c2 + c2_delta

        # c1 更新 (AAF 融合 c0->c1 与 c2->c1)
        c0_down_to_c1 = self.adapt_c0_to_c1(c0)           # (B,1024,192)
        c2_up_to_c1   = self.adapt_c2_to_c1(c2)           # (B,1024,192)
        B, L1, C1 = c0_down_to_c1.shape
        H1, W1 = self.c1_hw
        f0 = c0_down_to_c1.view(B,H1,W1,C1).permute(0,3,1,2)  # (B,192,32,32)
        f2 = c2_up_to_c1.view(B,H1,W1,C1).permute(0,3,1,2)
        fused_c1_feat = self.aaf_c1([f0, f2])             # (B,192,32,32)
        fused_c1_tokens = fused_c1_feat.flatten(2).transpose(1,2)
        c1_delta = mamba_c1(fused_c1_tokens)
        c1_next = c1 + c1_delta

        # c0 更新
        c1_up_to_c0 = self.adapt_c1_to_c0(c1)             # (B,4096,96)
        c0_delta = mamba_c0(c1_up_to_c0)
        c0_next = c0 + c0_delta

        return c0_next, c1_next, c2_next

    def forward(self, c0, c1, c2):
        # 第一次迭代
        c0_1, c1_1, c2_1 = self._fusion_step(
            c0, c1, c2, self.mamba_c0, self.mamba_c1, self.mamba_c2
        )
        # 第二次迭代（使用另一套 Mamba 堆栈）
        c0_2, c1_2, c2_2 = self._fusion_step(
            c0_1, c1_1, c2_1, self.mamba_c0_2, self.mamba_c1_2, self.mamba_c2_2
        )
        return c0_2, c1_2, c2_2

# ================== 6. 统一解码器 ==================
class UnifiedDecoderTokens(nn.Module):
    """
    输入：c0(4096,96), c1(1024,192), c2(256,768)
    输出：out (B,6,256,256)
    """
    def __init__(self):
        super().__init__()
        self.c0_proj = nn.Linear(96, 192)
        self.c2_proj = nn.Linear(768,192)
        self.fuse_conv1x1 = nn.Conv2d(192*3, 192, 1)

        # 转置卷积上采样 64->128->256
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(192,192,4,2,1),  # 64 -> 128
            nn.GELU(),
            nn.ConvTranspose2d(192,192,4,2,1),  # 128 -> 256
            nn.GELU(),
            nn.Conv2d(192,6,1)
        )

    def forward(self, c0, c1, c2, x_in):
        B = c0.shape[0]
        # c0: (B,4096,96) → proj
        c0_192 = self.c0_proj(c0)  # (B,4096,192)
        c0_map = c0_192.transpose(1,2).view(B,192,64,64)

        # c1: (B,1024,192) → upsample 32→64
        c1_map = c1.transpose(1,2).view(B,192,32,32)
        c1_map = F.interpolate(c1_map, size=(64,64), mode='bilinear', align_corners=False)

        # c2: (B,256,768) → upsample 16→64, proj
        c2_map = c2.transpose(1,2).view(B,768,16,16)
        c2_map = F.interpolate(c2_map, size=(64,64), mode='bilinear', align_corners=False)
        c2_map = self.c2_proj(c2_map.flatten(2).transpose(1,2)).transpose(1,2).view(B,192,64,64)

        fused = torch.cat([c0_map, c1_map, c2_map], dim=1)   # (B,576,64,64)
        fused = self.fuse_conv1x1(fused)                     # (B,192,64,64)
        D = self.decoder(fused)                              # (B,6,256,256)

        base = 0.3 * torch.cat([x_in, x_in], dim=1)          # (B,6,256,256)
        out = base + D
        return out

# ================== 7. 顶层整合主干（供 MTRRNet 调用） ==================
class MTRRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MultiBranchEncoder()
        self.fusion  = TwoIterMultiScaleFusion()
        self.decoder = UnifiedDecoderTokens()

    def forward(self, x):
        c0, c1, c2 = self.encoder(x)
        c0_f, c1_f, c2_f = self.fusion(c0, c1, c2)
        out = self.decoder(c0_f, c1_f, c2_f, x)
        return out
 

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
        self.out = self.netG_T(self.I) 
        self.out = self.netG_T(self.Ic) 
        self.fake_T, self.fake_R = self.out[:,0:3,:,:],self.out[:,3:6,:,:]
        self.c_map = torch.zeros_like(self.I)


        
 
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
                    msg = f"{layer_name:<70} | Mean: {mean:>15.6f} | Std: {std:>15.6f} | Shape: {tuple(output.shape)}"
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
                                f"Param: {name:<70} | "
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
