import torch

# 训练配置
def get_config():
    config = {
        'model_name': 'swin_base_patch4_window7_224',
        'data_dir': '/home/chenlb/EmojiClassifier/data',  # 数据根目录
        'result_dir': '/home/chenlb/EmojiClassifier/results/stat',
        'pretrained': True,
        'batch_size': 32,
        'num_workers': 8,
        'num_classes': 50,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'num_epochs': 100,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    return config