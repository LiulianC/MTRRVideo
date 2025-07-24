import math
import os.path
import os.path
import random
from os.path import join
import torch.utils.data as data
import cv2
import numpy as np
import torch.utils.data
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.signal import convolve2d
from scipy.stats import truncnorm
# from data.image_folder import make_dataset
# from data.torchdata import Dataset as BaseDataset
# from data.transforms import to_tensor
from torch.utils.data import Sampler

# 按目标宽度等比缩放图像，保持宽高比，调整高度为最接近的偶数
def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    h = math.ceil(h / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)

# 按目标高度等比缩放图像，保持宽高比，调整宽度为最接近的偶数。
def __scale_height(img, target_height):
    ow, oh = img.size
    if (oh == target_height):
        return img
    h = target_height
    w = int(target_height * ow / oh)
    w = math.ceil(w / 2.) * 2
    return img.resize((w, h), Image.BICUBIC)

# 对输入的两张图像（如退化图像与目标清晰图像）进行 ​​同步或异步的预处理与增强
def paired_data_transforms(img_1, img_2, img_3, unaligned_transforms=False):

    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    # 随机缩放​​：保持宽高比，缩放到 [320, 640] 之间的随机偶数尺寸
    target_size = int(random.randint(256, 640) / 2.) * 2
    ow, oh = img_1.size
    if ow >= oh:
        img_1 = __scale_height(img_1, target_size)
        img_2 = __scale_height(img_2, target_size)
        img_3 = __scale_height(img_3, target_size)
    else:
        img_1 = __scale_width(img_1, target_size)
        img_2 = __scale_width(img_2, target_size)
        img_3 = __scale_width(img_3, target_size)

    # ​​随机水平翻转​​（概率50%）
    if random.random() < 0.5:
        img_1 = TF.hflip(img_1)
        img_2 = TF.hflip(img_2)
        img_3 = TF.hflip(img_3)

    # 随机旋转​​（90°/180°/270°，概率50%）
    if random.random() < 0.5:
        angle = random.choice([90, 180, 270])
        img_1 = TF.rotate(img_1, angle)
        img_2 = TF.rotate(img_2, angle)
        img_3 = TF.rotate(img_3, angle)

    # 随机裁剪​​：随机在（i,j）位置会有小偏移

    i, j, h, w = get_params(img_1, (256, 256)) 
    img_1 = TF.crop(img_1, i, j, h, w) # 这里就已经变成目标大小了


    # 异步位移裁剪​​（若启用）：对第二张图像施加轻微的位置偏移
    if unaligned_transforms:
        # print('random shift')
        i_shift = random.randint(-10, 10)
        j_shift = random.randint(-10, 10)
        i += i_shift
        j += j_shift

    img_2 = TF.crop(img_2, i, j, h, w)


    # 异步位移裁剪​​（若启用）：对第二张图像施加轻微的位置偏移
    if unaligned_transforms:
        # print('random shift')
        i_shift = random.randint(-10, 10)
        j_shift = random.randint(-10, 10)
        i += i_shift
        j += j_shift

    img_3 = TF.crop(img_3, i, j, h, w)

    return img_1, img_2, img_3 # 三张图片一样大 




# ReflectionSynthesis 用于 ​​模拟真实场景中的反射效果合成​​
class ReflectionSynthesis(object):
    def __init__(self):
        # Kernel Size of the Gaussian Blurry
        self.kernel_sizes = [5, 7, 9, 11]
        self.kernel_probs = [0.1, 0.2, 0.3, 0.4]

        # Sigma of the Gaussian Blurry
        self.sigma_range = [2, 5]
        self.alpha_range = [0.8, 1.0]
        self.beta_range = [0.4, 1.0]

    def __call__(self, T_, R_):
        T_ = np.asarray(T_, np.float32) / 255.
        R_ = np.asarray(R_, np.float32) / 255.

        # 模拟真实反射的散射效应（如毛玻璃、水面波纹导致的模糊）。
        # 随机选择高斯核尺寸与标准差
        kernel_size = np.random.choice(self.kernel_sizes, p=self.kernel_probs)
        sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel2d = np.dot(kernel, kernel.T) # 生成二维高斯核
        # 对反射层每个通道进行卷积（模糊）
        for i in range(3):
            R_[..., i] = convolve2d(R_[..., i], kernel2d, mode='same')

        # 生成截断正态分布的随机系数（限制在合理范围）
        a1 = truncnorm((0.82 - 1.109) / 0.118, (1.42 - 1.109) / 0.118, loc=1.109, scale=0.118)
        a2 = truncnorm((0.85 - 1.106) / 0.115, (1.35 - 1.106) / 0.115, loc=1.106, scale=0.115)
        a3 = truncnorm((0.85 - 1.078) / 0.116, (1.31 - 1.078) / 0.116, loc=1.078, scale=0.116)
        
        # 调整透射层各通道强度
        T_[..., 0] *= a1.rvs()  
        T_[..., 1] *= a2.rvs()  
        T_[..., 2] *= a3.rvs()  
        # 反射强度控制​
        b = np.random.uniform(self.beta_range[0], self.beta_range[1])
        T, R = T_, b * R_ # # 反射层强度缩放
        
        if random.random() < 0.7:
            # # 光学叠加模型：I = T + R - T*R (避免过曝)
            I = T + R - T * R

        else:
            # 简单加法 + 自适应校正
            I = T + R
            if np.max(I) > 1:
                m = I[I > 1]
                m = (np.mean(m) - 1) * 1.3
                I = np.clip(T + np.clip(R - m, 0, 1), 0, 1)

        return T_, R_, I




# dataloader.reset() 需要dataset.reset()实现才能实现
class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, *args, **kwargs)
        self.shuffle = shuffle

    def reset(self):
        if self.shuffle:
            print('Reset Dataset...')
            self.dataset.reset()


# dataloader的抽样方法自定义
class CustomSampler(Sampler):
    def __init__(self, size1, size2, size3, samples_size1, samples_size2, samples_size3):
        self.size1 = size1
        self.size2 = size2
        self.size3 = size3
        self.samples_size1 = samples_size1
        self.samples_size2 = samples_size2
        self.samples_size3 = samples_size3

    def __iter__(self):
        # 生成合成风景的随机索引
        indices1 = torch.randperm(self.size1)[:self.samples_size1]
        # 生成合成血管的随机索引，并转换为全局索引
        indices2 = torch.randperm(self.size2)[:self.samples_size2] + self.samples_size1
        # 生成真实数据的随机索引，并转换为全局索引
        indices3 = torch.randperm(self.size3)[:self.samples_size3] + self.samples_size1 + self.samples_size2
        # 合并并打乱索引
        combined_indices = torch.cat([indices1, indices2, indices3])
        combined_indices = combined_indices[torch.randperm(len(combined_indices))]
        return iter(combined_indices.tolist())

    def __len__(self):
        return self.samples_size1 + self.samples_size2 + self.samples_size3












# size：限制加载的数据量
class DSRTestDataset(data.Dataset):
    def __init__(self, datadir, fns=None, size=None, enable_transforms=False, unaligned_transforms=False,
                 round_factor=1, flag=None, if_align=True, real=False, HW=[256,256]):
        super(DSRTestDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = [f for f in os.listdir(datadir) if f.endswith('-input.png')]
        self.enable_transforms = enable_transforms
        self.unaligned_transforms = unaligned_transforms
        self.round_factor = round_factor
        self.flag = flag
        self.if_align = True # if_align
        self.real = real
        self.h = HW[0]
        self.w = HW[1]
        
        
        self.I_paths = []
        self.R_paths = []
        self.T_paths = []
        for file in os.listdir(self.datadir):
            if file.endswith('-input.png') :
                self.I_paths.append(os.path.join(self.datadir, file))
                T = file.replace('-input.png', '-label1.png')
                self.T_paths.append((os.path.join(self.datadir, T)))
                R = file.replace('-input.png', '-label2.png')
                self.R_paths.append((os.path.join(self.datadir, R)))

        if size is not None and size<=len(self.I_paths): # 如果有size控制，那截取size个元素,而且是随机截取
            zipped = list(zip(self.I_paths,self.T_paths,self.R_paths))
            sampled_tuples = random.sample(zipped, size)
            self.I_paths_s,self.T_paths_s,self.R_paths_s=zip(*sampled_tuples)
        else:
            self.I_paths_s,self.T_paths_s,self.R_paths_s=self.I_paths,self.T_paths,self.R_paths

            

    def align(self, x1, x2, x3):
        h, w = self.h, self.w
        h, w = h // 32 * 32, w // 32 * 32
        # h_new, w = h + (32 - h % 32), w + 
        x1 = x1.resize((w, h))
        x2 = x2.resize((w, h))
        x3 = x3.resize((w, h))
        return x1, x2, x3

    def __getitem__(self, index):
        t_img = Image.open(self.T_paths_s[index]).convert('RGB')
        m_img = Image.open(self.I_paths_s[index]).convert('RGB')
        self.filename = str(self.fns[index])
        try:
            r_img = Image.fromarray(np.array(Image.open(self.R_paths_s[index]).convert('RGB'))*0.3) # 如果是合成反光图片 因为反光图片的反光场景是正常图像 亮度饱和度强 所以得压暗
        except Exception:
            r_img = Image.fromarray(np.clip((np.array(m_img, dtype=np.float32) - np.array(t_img, dtype=np.float32)), 0, 255).astype(np.uint8))

        if self.enable_transforms:
            t_img, m_img, r_img = paired_data_transforms(t_img, m_img, r_img, self.unaligned_transforms)

        if self.if_align:
            t_img, m_img, r_img = self.align(t_img, m_img, r_img)

        B = TF.to_tensor(t_img)
        M = TF.to_tensor(m_img)
        R = TF.to_tensor(r_img)

        dic = {'input': M, 'target_t': B, 'fn':self.filename, 'real': self.real, 'target_r': R}
        if self.flag is not None:
            dic.update(self.flag) # 用于将一个字典（或键值对序列）的内容合并到当前字典中
        return dic

    # 返回数据集的实际长度，受 size 参数限制
    def __len__(self):
        if self.size is not None:
            return min(len(self.I_paths), self.size)
        else:
            return len(self.I_paths)

    def reset(self):
        # 重新抽样 由于非常消耗cpu资源 现在更改成开头就做好随机抽样的index
        # if self.size is not None and self.size<=len(self.I_paths):
        #     zipped = list(zip(self.I_paths,self.T_paths,self.R_paths))
        #     sampled_tuples = random.sample(zipped, self.size)
        #     self.I_paths_s, self.T_paths_s, self.R_paths_s=zip(*sampled_tuples)
        pass





    









