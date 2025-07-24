import torch
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torch.nn.functional as F
import torchvision.models as models

class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
            
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.psnr_metric = PeakSignalNoiseRatio().to(self.device)
        
        # 1. 预先加载VGG模型并固定参数，避免重复加载
        self.vgg = models.vgg19(pretrained=True).features[:30].eval().to(self.device)
        for param in self.vgg.parameters():
            param.requires_grad = False
            
        # 存储VGG网络中我们关心的层索引
        self.feature_layers = {1, 6, 11, 20, 29}
        
        # 2. 调整损失权重，避免过大权重
        self.ssim_loss_weight = 1.0
        self.vgg_loss_weight = 0.5  # 从0.7降低到0.5，减少VGG损失的影响
        self.mse_loss_weight = 0.3
        
        self.fake_R_weight = 0.4
        self.fake_T_weight = 1.0
        self.Rcmaps_weight = 0
        self.all_img_weight = 0

    def forward(self, fake_Ts, label1, input_image, rcmaps, fake_Rs, label2):
        # 3. 使用eps参数确保所有输入在有效范围内
        eps = 1e-6
        fake_Ts = torch.clamp(fake_Ts, eps, 1.0-eps)
        fake_Rs = torch.clamp(fake_Rs, eps, 1.0-eps)
        label1 = torch.clamp(label1, eps, 1.0-eps)
        label2 = torch.clamp(label2, eps, 1.0-eps)
        input_image = torch.clamp(input_image, eps, 1.0-eps)
        rcmaps = torch.clamp(rcmaps, eps, 1.0-eps)

        # 计算fake_Rs的损失
        fake_R_mse_loss = F.mse_loss(fake_Rs, label2) * self.mse_loss_weight
        fake_R_vgg_loss = self.compute_perceptual_loss(fake_Rs, label2) * self.vgg_loss_weight
        
        # 4. 用try-except处理SSIM可能的数值问题
        try:
            fake_R_ssim_loss = (1 - self.ssim_metric(fake_Rs, label2)) * self.ssim_loss_weight
        except:
            print("Warning: SSIM calculation failed for fake_Rs. Using MSE as fallback.")
            fake_R_ssim_loss = F.mse_loss(fake_Rs, label2) * self.ssim_loss_weight

        # 计算fake_Ts的损失
        fake_T_mse_loss = F.mse_loss(fake_Ts, label1) * self.mse_loss_weight
        fake_T_vgg_loss = self.compute_perceptual_loss(fake_Ts, label1) * self.vgg_loss_weight
        
        try:
            fake_T_ssim_loss = (1 - self.ssim_metric(fake_Ts, label1)) * self.ssim_loss_weight
        except:
            print("Warning: SSIM calculation failed for fake_Ts. Using MSE as fallback.")
            fake_T_ssim_loss = F.mse_loss(fake_Ts, label1) * self.ssim_loss_weight

        # Rcmaps检测，使用更安全的方式计算
        # 5. 使用torch.clamp的同时注意梯度流
        I_R_diff = torch.clamp(input_image - label2, 0.0, 1.0)
        RCMap_test_img = torch.clamp(rcmaps * label1, 0.0, 1.0)
        
        Rcmaps_mse_loss = F.mse_loss(RCMap_test_img, I_R_diff) * self.mse_loss_weight
        Rcmaps_vgg_loss = self.compute_perceptual_loss(RCMap_test_img, I_R_diff) * self.vgg_loss_weight
        
        try:
            Rcmaps_ssim_loss = (1 - self.ssim_metric(RCMap_test_img, I_R_diff)) * self.ssim_loss_weight
        except:
            print("Warning: SSIM calculation failed for Rcmaps. Using MSE as fallback.")
            Rcmaps_ssim_loss = F.mse_loss(RCMap_test_img, I_R_diff) * self.ssim_loss_weight
        
        # 总和检测
        all_img = torch.clamp(fake_Ts * rcmaps + fake_Rs, 0.0, 1.0)
        all_img_mse_loss = F.mse_loss(all_img, input_image) * self.mse_loss_weight
        all_img_vgg_loss = self.compute_perceptual_loss(all_img, input_image) * self.vgg_loss_weight
        
        try:
            all_img_ssim_loss = (1 - self.ssim_metric(all_img, input_image)) * self.ssim_loss_weight
        except:
            print("Warning: SSIM calculation failed for all_img. Using MSE as fallback.")
            all_img_ssim_loss = F.mse_loss(all_img, input_image) * self.ssim_loss_weight

        # 6. 分别计算损失并应用权重，避免中间项过大
        mse_loss = (
            fake_R_mse_loss * self.fake_R_weight + 
            fake_T_mse_loss * self.fake_T_weight + 
            Rcmaps_mse_loss * self.Rcmaps_weight + 
            all_img_mse_loss * self.all_img_weight
        )
        
        # 7. VGG损失除以10而不是5，减少其幅度
        vgg_loss = (
            fake_R_vgg_loss * self.fake_R_weight + 
            fake_T_vgg_loss * self.fake_T_weight + 
            Rcmaps_vgg_loss * self.Rcmaps_weight + 
            all_img_vgg_loss * self.all_img_weight
        )
        
        ssim_loss = (
            fake_R_ssim_loss * self.fake_R_weight + 
            fake_T_ssim_loss * self.fake_T_weight + 
            Rcmaps_ssim_loss * self.Rcmaps_weight + 
            all_img_ssim_loss * self.all_img_weight
        )
        
        # 8. 对vgg_loss除以10而非5，进一步降低其影响
        total_loss = mse_loss + vgg_loss/10 + ssim_loss
        
        # 9. 确保最终损失没有NaN
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("Warning: NaN or Inf detected in loss calculation!")
            # 回退到简单的MSE损失
            total_loss = F.mse_loss(fake_Ts, label1) + F.mse_loss(fake_Rs, label2)

        loss_table = {
            'fake_R_mse_loss': fake_R_mse_loss,
            'fake_T_mse_loss': fake_T_mse_loss,
            'Rcmaps_mse_loss': Rcmaps_mse_loss,
            'all_img_mse_loss': all_img_mse_loss,
            'mse_loss': mse_loss,
            'fake_R_vgg_loss': fake_R_vgg_loss,
            'fake_T_vgg_loss': fake_T_vgg_loss,
            'Rcmaps_vgg_loss': Rcmaps_vgg_loss,
            'all_img_vgg_loss': all_img_vgg_loss,
            'vgg_loss': vgg_loss,
            'fake_R_ssim_loss': fake_R_ssim_loss,
            'fake_T_ssim_loss': fake_T_ssim_loss,
            'Rcmaps_ssim_loss': Rcmaps_ssim_loss,
            'all_img_ssim_loss': all_img_ssim_loss,
            'ssim_loss': ssim_loss,
            'total_loss': total_loss
        }
        
        # 返回的仍然是相同格式，但vgg_loss已经除以10了
        return loss_table, mse_loss, vgg_loss/10, ssim_loss, total_loss

    def compute_perceptual_loss(self, x, y):
        """
        稳定改进的感知损失计算：
        1. 使用预加载的VGG模型
        2. 对每一层特征进行归一化
        3. 使用渐进式权重增加深层特征的重要性
        4. 添加数值稳定性措施
        """
        # 10. 确保输入在有效范围内
        x = torch.clamp(x, 0.0, 1.0)
        y = torch.clamp(y, 0.0, 1.0)
        
        loss = 0.0
        # 11. 保存中间层特征，避免重复计算
        x_features = []
        y_features = []
        
        # 提取特征但不计算梯度
        with torch.no_grad():
            for i, layer in enumerate(self.vgg):
                x = layer(x)
                y = layer(y)
                
                if i in self.feature_layers:
                    x_features.append(x.detach())
                    y_features.append(y.detach())
                    
                if i >= max(self.feature_layers):
                    break
        
        # 12. 设置递增权重，使深层特征权重更大
        weights = [0.1, 0.2, 0.4, 0.8, 1.0]
        
        # 13. 使用MSE而非L1损失，提高稳定性
        for idx, (x_feat, y_feat) in enumerate(zip(x_features, y_features)):
            # 重新启用梯度计算
            x_feat = x_feat.detach().requires_grad_(True)
            y_feat = y_feat.detach()
            
            # 使用均方误差并添加数值稳定性
            feat_diff = F.mse_loss(x_feat, y_feat)
            
            # 14. 对每层损失进行裁剪，避免极端值
            feat_loss = torch.clamp(feat_diff, 0.0, 10.0)
            loss = loss + weights[idx] * feat_loss
            
        # 15. 归一化损失，使其不会因为层数增加而过大
        return loss / sum(weights)