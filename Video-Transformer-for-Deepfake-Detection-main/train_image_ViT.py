import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from models.imagetransformer import ImageTransformer
from models.DDFA import *
from utils_ViT import load_pretrained_weights, PRETRAINED_MODELS, as_tuple, resize_positional_embedding_
from transformers import ViTModel, ViTConfig  # 确保正确导入 transformers 包
from dataset_utils.training_dataset_creation import ImageTrainDataset


# 设置随机种子
seed = 17
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(seed)

# 数据路径
train_dir_real = r'C:\Users\Robin\Desktop\Video-Transformer-for-Deepfake-Detection-main\Video-Transformer-for-Deepfake-Detection-main\images\train_real'
train_dir_fake = r'C:\Users\Robin\Desktop\Video-Transformer-for-Deepfake-Detection-main\Video-Transformer-for-Deepfake-Detection-main\images\train_fake_FaceSwap'
train_dir_fake_2 = r'C:\Users\Robin\Desktop\Video-Transformer-for-Deepfake-Detection-main\Video-Transformer-for-Deepfake-Detection-main\images\train_fake_Face2Face'
train_dir_fake_3 = r'C:\Users\Robin\Desktop\Video-Transformer-for-Deepfake-Detection-main\Video-Transformer-for-Deepfake-Detection-main\images\train_fake_Deepfakes'
train_dir_fake_4 = r'C:\Users\Robin\Desktop\Video-Transformer-for-Deepfake-Detection-main\Video-Transformer-for-Deepfake-Detection-main\images\train_fake_NeuralTextures'

valid_dir_real = r'C:\Users\Robin\Desktop\Video-Transformer-for-Deepfake-Detection-main\Video-Transformer-for-Deepfake-Detection-main\images\valid_real'
valid_dir_fake = r'C:\Users\Robin\Desktop\Video-Transformer-for-Deepfake-Detection-main\Video-Transformer-for-Deepfake-Detection-main\images\valid_fake_FaceSwap'
valid_dir_fake_2 = r'C:\Users\Robin\Desktop\Video-Transformer-for-Deepfake-Detection-main\Video-Transformer-for-Deepfake-Detection-main\images\valid_fake_Face2Face'
valid_dir_fake_3 = r'C:\Users\Robin\Desktop\Video-Transformer-for-Deepfake-Detection-main\Video-Transformer-for-Deepfake-Detection-main\images\valid_fake_Deepfakes'
valid_dir_fake_4 = r'C:\Users\Robin\Desktop\Video-Transformer-for-Deepfake-Detection-main\Video-Transformer-for-Deepfake-Detection-main\images\valid_fake_NeuralTextures'

paths = [
    train_dir_real,
    train_dir_fake,
    train_dir_fake_2,
    train_dir_fake_3,
    train_dir_fake_4,
    valid_dir_real,
    valid_dir_fake,
    valid_dir_fake_2,
    valid_dir_fake_3,
    valid_dir_fake_4
]

# Debugging: Check if paths contain data
for path in paths:
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
    else:
        print(f"Path exists: {path}, Number of files: {len(os.listdir(path))}")

batch_size = 6
train_loader, valid_loader = ImageTrainDataset.get_image_batches(paths, batch_size)

# 模型初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageTransformer('B_16_imagenet1k', pretrained=True, image_size=300, num_classes=2,
                         seq_embed=True, hybrid=False, device=device)

# 训练参数
epochs = 15
lr = 3e-3

# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
# 学习率调度器
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# 训练循环
for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    for data, label in train_loader:
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)
            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)
    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )