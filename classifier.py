
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
import os
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim
import numpy as np
import argparse
from torch.utils.data import ConcatDataset



# PretrainedConvNext 类的核心功能是分类​
# 通过 self.head = nn.Linear(768, 6) 输出 ​​6 维向量​​，符合分类任务的典型设计（例如 6 类分类的 logits
# real data里 label1 和 input 的光线有差别 本模块用于调整各通道输出

# 调用语句 ipt = self.net_c(input_i)
class PretrainedConvNext(nn.Module):
    def __init__(self, model_name='convnext_base', pretrained=True):
        super(PretrainedConvNext, self).__init__()
        # Load the pretrained ConvNext model from timm
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0)# 直接调用timm库的模型 num_classes=0是取消库的线性分类层 
        self.head = nn.Linear(768, 6) # 自己加上线性6分类层
    def forward(self, x):
        with torch.no_grad():
            # 将输入张量 x 插值（缩放）到目标尺寸 (224, 224)，使用双线性插值（bilinear）方法，并启用 align_corners 对齐模式
            cls_input = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        
        # Forward pass through the ConvNext model
        out = self.model(cls_input)
        out = self.head(out)
        # 通过 nn.Linear 层进行线性变换，输出一个 6 维向量
        return out
    



class PretrainedConvNext_e2e(nn.Module):
    def __init__(self, model_name='convnext_base', pretrained=True):
        super(PretrainedConvNext_e2e, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.head = nn.Linear(768, 6)

    def forward(self, x):
        with torch.no_grad():
            cls_input = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        out = self.model(cls_input)
        out = self.head(out)
        alpha, beta = out[..., :3].unsqueeze(-1).unsqueeze(-1), out[..., 3:].unsqueeze(-1).unsqueeze(-1)
        out = alpha * x + beta
        return out



# ------------ SSIM LOSS ------------
def ssim_loss(pred, target):
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    B = pred_np.shape[0]
    total_ssim = 0.0
    for i in range(B):
        p = np.transpose(pred_np[i], (1, 2, 0))  # (H,W,C)
        t = np.transpose(target_np[i], (1, 2, 0))
        total_ssim += ssim(p, t, channel_axis=2, data_range=1.0, win_size=11)
    return 1 - (total_ssim / B)


# ------------ 可视化函数 ------------
def visualize(input, pred, target, epoch, save_dir='./cls/vis'):
    os.makedirs(save_dir, exist_ok=True)
    input = torch.clamp(input, 0, 1)
    pred = torch.clamp(pred, 0, 1)
    target = torch.clamp(target, 0, 1)
    for i in range((pred.size(0))):  # 保存前4个样本
        grid = torch.cat([input[i], pred[i], target[i]], dim=-1)  # 横向拼接
        save_image(grid, os.path.join(save_dir, f'epoch_{epoch}_sample_{i}.png'))


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
opts = parser.parse_args()
opts.batch_size = 32
opts.sampler_size1 = 300
opts.sampler_size2 = 0
opts.sampler_size3 = 800
opts.epochs = 1
opts.model_path='./cls/cls_models/latest.pth'  
# opts.model_path=None  
current_lr = 1e-5
opts.num_workers = 0

# ------------ 主训练逻辑 ------------
if __name__ == "__main__":
    from dataset.new_dataset1 import *  # 你需要实现 train_loader, val_loader 的模块

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PretrainedConvNext_e2e('convnext_small_in22k').to(device)
    if opts.model_path is not None:
        print(f"Loading model from {opts.model_path}")
        model.load_state_dict(torch.load(opts.model_path, map_location=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)
    mse_fn = nn.MSELoss()

    # 数据集：你需要提供 train_loader 和 val_loader
    fit_datadir = '/home/gzm/gzm-RDNet1/dataset/laparoscope_gen'
    fit_data = DSRTestDataset(datadir=fit_datadir, fns='/home/gzm/gzm-RDNet1/dataset//laparoscope_gen_index/train1.txt',size=opts.sampler_size1, enable_transforms=False,if_align=True,real=False, HW=[256,256])

    tissue_gen = './data/tissue_gen'
    tissue_gen_data = DSRTestDataset(datadir=tissue_gen, fns='./data/tissue_gen_index/train1.txt',size=opts.sampler_size2, enable_transforms=False,if_align=True,real=False, HW=[256,256])

    tissue_dir = './data/tissue_real'
    tissue_data = DSRTestDataset(datadir=tissue_dir,fns='./data/tissue_real_index/train1.txt',size=opts.sampler_size3, enable_transforms=True,if_align=True,real=False, HW=[256,256])

    # 使用ConcatDataset方法合成数据集 能自动跳过空数据集
    train_data = ConcatDataset([fit_data, tissue_gen_data, tissue_data])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=opts.batch_size, shuffle=True, num_workers = opts.num_workers, drop_last=False, pin_memory=True)



    test_data_dir1 = './data/tissue_real'
    test_data1 = DSRTestDataset(datadir=test_data_dir1, fns='./data/tissue_real_index/eval1.txt', enable_transforms=False, if_align=True, real=True, HW=[256,256], size=200)

    test_data_dir2 = './data/hyperK_000'
    test_data2 = TestDataset(datadir=test_data_dir2, fns='./data/hyperK_000_list.txt', enable_transforms=False, if_align=True, real=True, HW=[256,256], size=200)

    test_data_dir3 = '/home/gzm/gzm-RDNet1/dataset/laparoscope_gen'
    test_data3 = DSRTestDataset(datadir=test_data_dir3, fns='/home/gzm/gzm-RDNet1/dataset/laparoscope_gen_index/eval1.txt', enable_transforms=False, if_align=True, real=True, HW=[256,256], size=200)
    test_data = ConcatDataset([test_data1, test_data2, test_data3])

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers, drop_last=False, pin_memory=True)

    best_val_loss = float('inf')
    num_epochs = opts.epochs

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        model.train()
        running_loss = 0.0
        with tqdm(train_loader, desc=f"[Train] Epoch {epoch}") as pbar:
            for batch in pbar:
                inputs, labels, fns = batch['input'], batch['target_t'], batch['fn']
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                preds = model(inputs)

                mse = mse_fn(preds, labels)
                ssim_l = ssim_loss(preds, labels)
                loss = mse + ssim_l

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({'loss': loss.item(), 'ssiml': ssim_l.item(), 'mse': mse.item()})

        # ----------- 验证阶段 -----------
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        all_inputs = []
        with torch.no_grad():
            print(f"Validating at epoch {epoch}")
            with tqdm(test_loader, desc=f"[Val] Epoch {epoch}") as pbar:
                for batch in pbar:
                    inputs, labels, fns = batch['input'], batch['target_t'], batch['fn']
                    inputs, labels = inputs.to(device), labels.to(device)

                    preds = model(inputs)

                    mse = mse_fn(preds, labels)
                    ssim_l = ssim_loss(preds, labels)
                    loss = mse + ssim_l
                    val_loss += loss.item()

                    all_inputs.append(inputs)
                    all_preds.append(preds)
                    all_labels.append(labels)

        val_loss /= len(test_loader)

        # 可视化
        visualize(torch.cat(all_inputs)[:], torch.cat(all_preds)[:], torch.cat(all_labels)[:], epoch)

        # 保存模型
        os.makedirs('./cls/cls_models', exist_ok=True)
        torch.save(model.state_dict(), "./cls/cls_models/latest.pth")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best model at epoch {epoch} with loss {best_val_loss:.4f}")
            torch.save(model.state_dict(), os.path.join('./cls/cls_models',f"model_{epoch}.pth"))
        else:
            print(f"Epoch {epoch} did not improve. Best loss:{best_val_loss:.4f}  now: {val_loss:.4f}")

# 模型各种分类
# 模型名称 (timm)	    参数量	输入分辨率	预训练数据集
# convnext_tiny	        28M	    224x224	    ImageNet-1k
# convnext_small	    50M	    224x224	    ImageNet-1k
# convnext_small_in22k	50M	    224x224	    ImageNet-22k
# convnext_base_in22k	88M	    224x224	    ImageNet-22k

# ConvNeXt_Block(
#     # 深度可分离卷积（大核卷积）
#     nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim),  # 7×7 组卷积
#     # 层归一化（LayerNorm）
#     nn.LayerNorm(dim),
#     # 通道扩展与收缩（类似 MLP）
#     nn.Linear(dim, 4 * dim),  # 扩展
#     nn.GELU(),                # 激活
#     nn.Linear(4 * dim, dim),  # 收缩
#     # 可选：DropPath 正则化
#     DropPath(p=drop_rate)
# )