import torch
import torch.nn as nn

def monitor_layer_stats(model, input_data):
    """为模型每一层注册前向钩子，打印输出张量的均值和标准差"""
    hooks = []
    
    # 定义钩子函数
    def hook_fn(module, input, output, layer_name):
        if isinstance(output, torch.Tensor):
            mean = output.mean().item()
            std = output.std().item()
            print(f"Layer: {layer_name:20} | Mean: {mean:8.4f} | Std: {std:8.4f} | Shape: {tuple(output.shape)}")
    
    # 遍历所有子模块并注册钩子
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList):  # 过滤容器类（如Sequential）
            hook = module.register_forward_hook(
                lambda m, inp, out, name=name: hook_fn(m, inp, out, name)
            )
            hooks.append(hook)
    
    # 运行前向传播触发钩子
    with torch.no_grad():
        model(input_data)
    
    # 移除钩子（避免内存泄漏）
    for hook in hooks:
        hook.remove()