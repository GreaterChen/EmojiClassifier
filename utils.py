import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import os

def plot_training_progress(history, save_path='training_progress.png'):
    """
    绘制训练进度图
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 绘制损失曲线
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Training and Validation Loss')
    
    # 绘制准确率曲线
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def set_seed(seed=42):
    """
    固定所有可能的随机数种子，确保结果可复现
    
    Args:
        seed (int): 随机数种子，默认为42
    """
    # Python随机数生成器
    random.seed(seed)
    
    # Numpy随机数生成器
    np.random.seed(seed)
    
    # PyTorch随机数生成器
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU的情况
    
    # PyTorch后端
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False     # 禁用cudnn的自动调优功能
    
    # 设置Python哈希种子
    os.environ['PYTHONHASHSEED'] = str(seed)


# 使用示例
def get_model_names():
    """获取所有支持的模型名称"""
    base_models = [
        'resnet50', 'resnet101', 'resnet152',
        'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
        'swin_base_patch4_window7_224', 'swin_small_patch4_window7_224'
    ]
    return base_models