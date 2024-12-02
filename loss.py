import torch
import torch.nn as nn
import torch.nn.functional as F

# 损失函数
class CombinedLoss(nn.Module):
    def __init__(self, temp=0.07):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.style_contrast = StyleContrastLoss(temp)
        
    def forward(self, outputs, style_features, labels):
        # 分类损失
        ce_loss = self.ce_loss(outputs, labels)
        
        # 风格对比损失
        style_loss = self.style_contrast(style_features, labels)
        
        # 可以调整权重
        return ce_loss + 0.1 * style_loss

class StyleContrastLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        similarity = torch.mm(features, features.t()) / self.temperature
        masks = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        
        # 去除对角线
        masks = masks - torch.eye(masks.shape[0], device=masks.device)
        
        # 计算正样本对的损失
        pos_loss = -torch.log(
            (similarity * masks).sum(1) / 
            (torch.exp(similarity) * (1 - torch.eye(similarity.shape[0], device=similarity.device))).sum(1)
        )
        
        return pos_loss.mean()