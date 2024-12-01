import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet101, resnet152, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights
from timm import create_model

class EmojiClassifier(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=50, pretrained=True):
        super(EmojiClassifier, self).__init__()
        
        self.model_name = model_name.lower()
        
        # 模型配置字典
        self.model_configs = {
            # ResNet 系列
            'resnet50': {
                'model': resnet50,
                'weights': ResNet50_Weights.IMAGENET1K_V2 if pretrained else None,
                'features': 2048
            },
            'resnet101': {
                'model': resnet101,
                'weights': ResNet101_Weights.IMAGENET1K_V2 if pretrained else None,
                'features': 2048
            },
            'resnet152': {
                'model': resnet152,
                'weights': ResNet152_Weights.IMAGENET1K_V2 if pretrained else None,
                'features': 2048
            },
            # EfficientNet 系列
            'efficientnet_b0': {
                'model': efficientnet_b0,
                'weights': EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None,
                'features': 1280
            },
            'efficientnet_b1': {
                'model': efficientnet_b1,
                'weights': EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None,
                'features': 1280
            },
            'efficientnet_b2': {
                'model': efficientnet_b2,
                'weights': EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None,
                'features': 1408
            },
        }
        
        # 初始化backbone
        if self.model_name in self.model_configs:
            config = self.model_configs[self.model_name]
            self.backbone = config['model'](weights=config['weights'])
            n_features = config['features']
            
            # 移除原始分类头
            if 'resnet' in self.model_name:
                self.backbone.fc = nn.Identity()
            elif 'efficientnet' in self.model_name:
                self.backbone.classifier = nn.Identity()
                
        # Swin Transformer 系列
        elif 'swin' in self.model_name:
            self.backbone = create_model(
                self.model_name,
                pretrained=pretrained,
                num_classes=0
            )
            n_features = self.backbone.num_features
            
        else:
            raise ValueError(f"Model {model_name} not supported")
        
        # 统一的分类头设计
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
    
    def load_pretrained(self, path):
        """加载预训练权重"""
        checkpoint = torch.load(path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.load_state_dict(checkpoint)
            
    def get_model_info(self):
        """获取模型信息"""
        return {
            'model_name': self.model_name,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

# 使用示例
def get_model_names():
    """获取所有支持的模型名称"""
    base_models = [
        'resnet50', 'resnet101', 'resnet152',
        'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
        'swin_base_patch4_window7_224', 'swin_small_patch4_window7_224'
    ]
    return base_models