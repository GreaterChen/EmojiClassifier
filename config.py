import torch

# 训练配置
def get_config():
    config = {
        'model_name': 'swin_base_patch4_window7_224',
        'data_dir': '/home/chenlb/EmojiClassifier/data',  # 数据根目录
        'result_dir': '/home/chenlb/EmojiClassifier/results/all',
        'pretrained': True,
        'batch_size': 2,
        'num_workers': 4,
        'num_classes': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'num_epochs': 300,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    return config