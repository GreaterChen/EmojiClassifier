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
        # L2归一化
        features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        similarity = torch.mm(features, features.t()) / self.temperature  # (batch_size, batch_size)
        
        # 创建标签掩码
        masks = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # (batch_size, batch_size)
        
        # 去除自身对比
        eye = torch.eye(features.size(0), device=features.device)
        masks = masks - eye
        
        # 为了数值稳定性，使用logsumexp
        exp_sim = torch.exp(similarity - torch.max(similarity, dim=1, keepdim=True)[0])
        
        # 计算分母 (所有负样本的exp sum)
        neg_mask = 1 - eye
        denominator = torch.sum(exp_sim * neg_mask, dim=1)
        
        # 计算分子 (正样本的exp sum)
        numerator = torch.sum(exp_sim * masks, dim=1)
        
        # 避免除零
        eps = 1e-8
        loss = -torch.log(numerator / (denominator + eps) + eps)
        
        # 确保有正样本的样本才计算损失
        valid_mask = (masks.sum(1) > 0)
        if valid_mask.sum() > 0:
            loss = loss[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=features.device)
            
        return loss