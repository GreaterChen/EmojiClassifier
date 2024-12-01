import torch

# 训练配置
def get_config():
    config = {
        'data_dir': '/home/chenlb/machineLearning/data',  # 数据根目录
        'result_dir': '/home/chenlb/machineLearning/results/backbone',
        'batch_size': 32,
        'num_workers': 4,
        'num_classes': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'num_epochs': 30,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    return config