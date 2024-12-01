import matplotlib.pyplot as plt

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