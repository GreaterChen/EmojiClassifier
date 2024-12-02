import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101, resnet152, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights
from timm import create_model

class EmojiClassifier(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=50, pretrained=True):
        super(EmojiClassifier, self).__init__()
        
        self.model_name = model_name.lower()
        
        # 1. 构建主干网络和基础分类器
        n_features = self.build_encoder(model_name, pretrained)
        self.build_classifier(1280, num_classes)
        
        # 2. 构建风格感知分支
        self.style_branch = StyleBranch(3)
        
        # 3. 特征融合模块
        self.feature_fusion = FeatureFusion(n_features)
    
    def forward(self, x):
        # 1. 提取基础特征
        features = self.backbone(x) # (B, 1024)
        
        # 2. 风格特征提取
        style_features = self.style_branch(x)   # (B, 1024)
        
        # 3. 特征融合
        fused_features = self.feature_fusion(features, style_features)
        
        # 4. 分类
        output = self.classifier(fused_features)
        
        if self.training:
            return output, style_features
        return output
    
    def load_pretrained(self, path):
        """加载预训练权重"""
        checkpoint = torch.load(path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.load_state_dict(checkpoint)

    def build_encoder(self, model_name='resnet50', pretrained=True):
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
        
        return n_features
    
    def build_classifier(self, n_features, num_classes=50):
        # 统一的分类头设计
        self.classifier = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
            
    def get_model_info(self):
        """获取模型信息"""
        return {
            'model_name': self.model_name,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    

class StyleBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        # 扩展到32通道
        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # 1. 统计特征提取器
        self.stat_extractor = AdaptiveStatExtractor(32)  # 输出维度: 32*4 = 128
        
        # 2. 纹理编码器
        # self.texture_encoder = MultiScaleTextureEncoder(32)  # 输出维度: 32*32*3 = 3072
        
        # 特征融合网络
        self.fusion = nn.Sequential(
            # 分别处理两种特征
            FusionBlock(128, 256),    # 统计特征处理 128->512
            FusionBlock(3072, 512),   # 纹理特征处理 3072->512
            
            # 最终融合得到1024维特征
            # nn.Linear(256, 256),    # 连接后的维度(512+512=1024)
            nn.LayerNorm(256),
        )
        
    def forward(self, x):
        # 扩展特征通道
        x = self.expand_conv(x)
        
        # 1. 提取统计特征和纹理特征
        stat_features = self.stat_extractor(x)      # (batch_size, 128)
        # texture_features = self.texture_encoder(x)   # (batch_size, 3072)
        
        # 2. 分别处理每种特征
        stat_processed = self.fusion[0](stat_features)    # (batch_size, 512)
        # texture_processed = self.fusion[1](texture_features)  # (batch_size, 512)
        
        # 3. 组合并最终融合
        # combined = torch.cat([stat_processed, texture_processed], dim=1)  # (batch_size, 1024)
        return self.fusion[-1](stat_processed)  # (batch_size, 1024)
    
class FusionBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        hidden_dim = min(max(in_dim // 2, out_dim), 1024)
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x)

class AdaptiveStatExtractor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels, affine=False)
        
        # 自适应权重网络
        self.weight_net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        # 计算统计量
        mean = x.mean([2, 3])  # (batch_size, channels)
        std = x.std([2, 3])    # (batch_size, channels)
        
        # 正确展开mean以进行广播
        mean_expanded = mean.unsqueeze(-1).unsqueeze(-1)  # (batch_size, channels, 1, 1)
        
        # 计算高阶矩
        skewness = torch.pow(x - mean_expanded, 3).mean([2, 3]) # (batch_size, channels)
        kurtosis = torch.pow(x - mean_expanded, 4).mean([2, 3]) # (batch_size, channels)
        
        # 对每个通道的统计量进行组合
        stats = torch.stack([mean, std, skewness, kurtosis], dim=2)  # (batch_size, channels, 4)
        
        # 计算每个统计量的权重 (对所有通道取平均)
        weights = self.weight_net(stats.mean(dim=1))  # (batch_size, 4)
        
        # 应用权重
        weighted_stats = stats * weights.unsqueeze(1)  # (batch_size, channels, 4)
        
        return weighted_stats.flatten(1)  # (batch_size, channels*4)

class MultiScaleTextureEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scales = [1, 2, 4]
        
        # 多尺度卷积
        self.convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=1, dilation=s)
            for s in self.scales
        ])
        
    def compute_gram(self, x):
        b, c, h, w = x.size()
        features = x.view(b, c, -1)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(c * h * w)
    
    def forward(self, x):
        # 多尺度特征提取
        multi_scale_features = []
        for conv in self.convs:
            features = conv(x)
            gram = self.compute_gram(features)
            multi_scale_features.append(gram)
        
        # 融合多尺度特征
        return torch.cat([f.flatten(1) for f in multi_scale_features], dim=1)

class StyleAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        attention_weights = self.attention(x)
        out = x * attention_weights
        return out.view(x.size(0), -1)  # 展平为(batch_size, channels*h*w)

class FeatureFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(1024+256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Sigmoid()
        )
        
    def forward(self, content_features, style_features):
        combined = torch.cat([content_features, style_features], dim=1)
        # attention_weights = self.attention(combined)
        # return content_features * attention_weights + style_features * (1 - attention_weights)
        return combined


