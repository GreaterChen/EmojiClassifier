import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from timm import create_model
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import os
from PIL import Image
from tqdm import tqdm

# 优化transform以减少PIL Image转换
def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Lambda(lambda x: x if isinstance(x, Image.Image) else Image.fromarray(x)),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Lambda(lambda x: x if isinstance(x, Image.Image) else Image.fromarray(x)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    # 创建训练集数据集实例
    train_dataset = EmojiDataset(
        root_dir=data_dir,
        transform=get_transforms(is_train=True),
        mode='train'
    )
    
    # 从训练集中分割出验证集
    train_indices, val_indices = train_test_split(
        range(len(train_dataset)),
        test_size=0.1,
        random_state=42,
        stratify=train_dataset.labels
    )
    
    # 创建训练集和验证集的采样器
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    # 创建训练和验证数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 创建测试集的数据集和数据加载器
    test_dataset = EmojiDataset(
        root_dir=data_dir,
        transform=get_transforms(is_train=False),
        mode='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.get_classes()

# 创建数据集类
class EmojiDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.images = []
        self.labels = []
        
        if mode == 'train':
            # 获取所有类别
            self.classes = sorted(os.listdir(os.path.join(root_dir, 'train')))
            self.classes = [item.replace(" ", ",") for item in self.classes]
                
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
            
            # 收集所有训练图像路径
            image_paths = []
            labels = []
            for class_name in self.classes:
                class_dir = os.path.join(root_dir, 'train', class_name)
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.png'):
                        image_paths.append(os.path.join(class_dir, img_name))
                        labels.append(self.class_to_idx[class_name])
            
            # 预加载所有图像到内存
            print(f"Loading {len(image_paths)} training images into memory...")
            for img_path in tqdm(image_paths):
                # 读取图像并转换为RGB
                img = Image.open(img_path).convert('RGB')
                # 转换为numpy数组存储，节省内存
                img_array = np.array(img)
                self.images.append(img_array)
            self.labels = labels
            
        else:
            # 测试集直接读取所有png文件
            test_dir = os.path.join(root_dir, 'test')
            image_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                         if f.endswith('.png')]
            image_paths.sort()  # 确保文件顺序一致
            
            # 预加载测试集图像
            print(f"Loading {len(image_paths)} test images into memory...")
            self.image_names = [os.path.basename(p) for p in image_paths]
            for img_path in tqdm(image_paths):
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img)
                self.images.append(img_array)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 从numpy数组转换回PIL Image
        image = Image.fromarray(self.images[idx])

        if self.transform:
            image = self.transform(image)
            
        if self.mode == 'train':
            return image, self.labels[idx]
        else:
            return image, self.image_names[idx]
    
    def get_classes(self):
        return self.classes