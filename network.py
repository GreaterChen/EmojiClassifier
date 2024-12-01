from timm import create_model
import torch.nn as nn


class EmojiClassifier(nn.Module):
    def __init__(self, num_classes=50, pretrained=True):
        super(EmojiClassifier, self).__init__()
        
        # 加载预训练的Swin Transformer模型
        self.backbone = create_model(
            'swin_base_patch4_window7_224',
            pretrained=pretrained,
            num_classes=0  # 移除原始分类头
        )
        
        # 获取backbone的输出特征维度
        n_features = self.backbone.num_features
        
        # 添加分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output
