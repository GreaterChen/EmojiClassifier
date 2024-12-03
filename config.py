import torch

# 训练配置
def get_config():
    config = {
        'model_name': 'swin_base_patch4_window7_224',
        # 'model_name': 'swin_base_patch4_window7_224.ms_in22k_ft_in1k',
        'data_dir': '/home/chenlb/EmojiClassifier/data',  # 数据根目录
        'result_dir': '/home/chenlb/EmojiClassifier/results/stat_2_b128',
        'pretrained': True,
        'batch_size': 64,
        'num_workers': 8,
        'num_classes': 50,
        'learning_rate': 8e-4,
        'weight_decay': 1e-4,
        'num_epochs': 200,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    return config