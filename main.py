import torch
from torchvision import transforms
from collections import defaultdict
from dataset import *
from config import *
from network import *
from utils import *
# 训练函数
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total

# 验证函数
@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total

# 添加测试集预测函数
@torch.no_grad()
def predict_test(model, test_loader, device, classes):
    model.eval()
    predictions = []
    filenames = []
    
    for images, image_names in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        
        # 将预测的类别索引转换为类别名称
        predicted_classes = [classes[idx] for idx in predicted.cpu().numpy()]
        predictions.extend(predicted_classes)
        filenames.extend(image_names)
    
    # 创建提交文件
    with open('submission.csv', 'w') as f:
        f.write('file_name,predicted_class\n')
        for fname, pred in zip(filenames, predictions):
            f.write(f'{fname},{pred}\n')


# 使用示例
def main():
    config = get_config()
    os.makedirs(config['result_dir'], exist_ok=True)
    
    # 获取数据加载器和类别信息
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        config['data_dir'],
        config['batch_size'],
        config['num_workers']
    )
    
    # 创建模型
    model = EmojiClassifier(num_classes=len(classes)).to(config['device'])
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs']
    )
    
    # 打印数据集信息
    print(f"Number of classes: {len(classes)}")
    print(f"Classes: {classes}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # 开始训练...
    history = defaultdict(list)
    best_acc = 0
    for epoch in range(config['num_epochs']):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config['device']
        )
        
        val_loss, val_acc = validate(
            model, val_loader, criterion, config['device']
        )
        
        scheduler.step()

        # 记录训练历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'classes': classes
            }, os.path.join(config['result_dir'], f'best_model.pth'))
            
        print(f'Epoch [{epoch+1}/{config["num_epochs"]}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # 绘制并保存训练进度图
        plot_training_progress(history, save_path=os.path.join(config['result_dir'], 'training_progress.png'))

        # 训练完成后进行测试集预测
    print("Predicting test set...")
    predict_test(model, test_loader, config['device'], classes)
    print("Predictions saved to submission.csv")

if __name__ == '__main__':
    main()