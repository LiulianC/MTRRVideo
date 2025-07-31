import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)              # 对当前设备
    torch.cuda.manual_seed_all(seed)          # 对所有设备

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    # 避免使用非确定性算法