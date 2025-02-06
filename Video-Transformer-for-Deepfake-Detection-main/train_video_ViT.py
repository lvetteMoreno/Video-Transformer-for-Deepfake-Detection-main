import random, torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


#Import ViT Packages
from models.imagetransformer import ImageTransformer
from models.videotransformer import VideoTransformer
from models.DDFA import *
from utils_ViT import load_pretrained_weights, PRETRAINED_MODELS, as_tuple, resize_positional_embedding_
from models.transformer import *
from dataset_utils.training_dataset_creation import VideoTrainDataset


# Import 3DDFA Packages
import yaml
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
import warnings
import re

from LipForensics.models.spatiotemporal_net import get_model
model1 = get_model()


print(torch.cuda.is_available())
def ignore_onnxruntime_warnings(message, category, filename, lineno, file=None, line=None):
    if isinstance(message, UserWarning) and "Expected shape from model" in str(message):
        return None
    return f"{message} {category} {filename}:{lineno}"

# 设置自定义警告格式化函数
warnings.formatwarning = ignore_onnxruntime_warnings

seed = 17
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

paths = []
# 数据路径
train_dir_real = r"D:\FF++\FaceForensics++\FaceForensics++\manipulated_sequences\DeepFakeDetection\c23\videos"
train_dir_fake = r"D:\FF++\FaceForensics++\FaceForensics++\manipulated_sequences\FaceSwap\c23\videos"
train_dir_fake_2 = r"D:\FF++\FaceForensics++\FaceForensics++\manipulated_sequences\Face2Face\c23\videos"
train_dir_fake_3 = r"D:\FF++\FaceForensics++\FaceForensics++\manipulated_sequences\Deepfakes\c23\videos"
train_dir_fake_4 = r"D:\FF++\FaceForensics++\FaceForensics++\manipulated_sequences\NeuralTextures\c23\videos"

valid_dir_real = r"D:\FF++\FaceForensics++\FaceForensics++\original_sequences\actors\c23\videos"
valid_dir_fake = r"D:\FF++\FaceForensics++\FaceForensics++\original_sequences\actors\c23\videos"
valid_dir_fake_2 = r"D:\FF++\FaceForensics++\FaceForensics++\original_sequences\actors\c23\videos"
valid_dir_fake_3 = r"D:\FF++\FaceForensics++\FaceForensics++\original_sequences\actors\c23\videos"
valid_dir_fake_4 = r"D:\FF++\FaceForensics++\FaceForensics++\original_sequences\actors\c23\videos"


paths.append(train_dir_real)
paths.append(train_dir_fake)
paths.append(train_dir_fake_2)
paths.append(train_dir_fake_3)
paths.append(train_dir_fake_4)
paths.append(valid_dir_real)
paths.append(valid_dir_fake)
paths.append(valid_dir_fake_2)
paths.append(valid_dir_fake_3)
paths.append(valid_dir_fake_4)

batch_size = 1
train_loader, valid_loader = VideoTrainDataset.get_video_batches(paths, batch_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = VideoTransformer('B_16_imagenet1k', pretrained=True, image_size = 384, num_classes = 2,
                        seq_embed=True, hybrid=True, variant='video', device=device)

epochs = 15
lr = 3e-3
# gamma = 0.7

# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
# scheduler
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    for data, label in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):  # 添加进度条
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
        for data, label in (valid_loader):
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

