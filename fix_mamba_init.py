import torch
import torch.nn as nn
from MTRRNet import MTRREngine

def improved_init_weights(m):
    """改进的权重初始化函数，特别针对深度网络和Mamba模块"""
    
    # 通用卷积层 - 使用Xavier初始化以保持方差稳定
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        # 对于深度网络，Xavier初始化更稳定
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    
    # 线性层 - 使用Xavier初始化
    elif isinstance(m, nn.Linear):
        # 检查是否是Mamba的投影层
        if hasattr(m, '__module__') and 'mamba' in m.__module__.lower():
            # 对Mamba的线性层使用较小的初始化
            nn.init.xavier_uniform_(m.weight, gain=0.5)
        else:
            nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    
    # LayerNorm和BatchNorm - 标准初始化
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
    
    # PReLU特殊初始化
    elif isinstance(m, nn.PReLU):
        nn.init.uniform_(m.weight, 0.1, 0.2)  # 稍微增加初始值
    
    # 针对自定义模块的参数
    for name, param in m.named_parameters(recurse=False):
        # Mamba特定参数
        if 'dt_proj' in name or 'x_proj' in name or 'out_proj' in name:
            # 对Mamba的关键投影层使用较小的初始化
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param, gain=0.5)
            else:
                nn.init.zeros_(param)
        # A_log参数（Mamba的状态转移矩阵）
        elif 'A_log' in name:
            # 保持默认初始化或使用较小的负值
            pass
        # D参数（Mamba的直接连接）
        elif name == 'D' and param.dim() == 1:
            # 初始化为较小的正值，避免过度缩放
            nn.init.uniform_(param, 0.5, 1.0)

def apply_improved_init(model):
    """应用改进的初始化到整个模型"""
    model.apply(improved_init_weights)
    
    # 特别处理PatchEmbed
    for name, module in model.named_modules():
        if 'patch_embed' in name.lower():
            # 对PatchEmbed使用较大的初始化，避免数值过小
            if hasattr(module, 'proj'):
                if hasattr(module.proj, 'weight'):
                    nn.init.xavier_uniform_(module.proj.weight, gain=2.0)
                if hasattr(module.proj, 'bias') and module.proj.bias is not None:
                    nn.init.zeros_(module.proj.bias)
    
    print("Applied improved initialization")
    return model

if __name__ == "__main__":
    # 测试初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class DummyOpts:
        def __init__(self):
            self.model_path = None
    
    opts = DummyOpts()
    model = MTRREngine(opts, device)
    
    # 应用改进的初始化
    model.netG_T = apply_improved_init(model.netG_T)
    
    # 测试前向传播
    test_input = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        rmap, out = model.netG_T(test_input)
        print(f"Output mean: {out.mean().item():.6f}, std: {out.std().item():.6f}")
        print(f"Rmap mean: {rmap.mean().item():.6f}, std: {rmap.std().item():.6f}")