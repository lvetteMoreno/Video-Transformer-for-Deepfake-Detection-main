import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.spatiotemporal_net import get_model
from data_process.transform import ToTensorMouth, NormalizeMouth
from torch.nn.utils.rnn import pad_sequence
import argparse
from tqdm import tqdm  # 导入 tqdm 库

# 数据集类
class DeepfakeDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.video_paths = []
        self.labels = []

        for label in os.listdir(data_folder):
            label_folder = os.path.join(data_folder, label)
            if os.path.isdir(label_folder):
                for filename in os.listdir(label_folder):
                    if filename.endswith(".npy"):
                        self.video_paths.append(os.path.join(label_folder, filename))
                        self.labels.append(1.0 if label == "fake" else 0.0)  # 使用浮点数标签

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        video_data = np.load(video_path)
        video_data = torch.from_numpy(video_data).float()

        if self.transform:
            video_data = self.transform(video_data)

        length = video_data.shape[0]

        return video_data, label, length

# 自定义 collate_fn
def collate_fn(batch):
    videos, labels, lengths = zip(*batch)
    videos = pad_sequence(videos, batch_first=True, padding_value=0)
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float)  # 使用浮点数标签
    return videos, labels, lengths

# 数据转换
mean = (0.5,)
std = (0.5,)
transform = transforms.Compose([
    ToTensorMouth(),
    NormalizeMouth(mean, std)
])

# 添加命令行参数解析
parser = argparse.ArgumentParser(description="Deepfake Detection Training Script")
parser.add_argument("--data_folder", type=str, default="data_process/dataset/output", help="Path to the dataset folder")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train")
parser.add_argument("--weights_folder", type=str, default="models/weights", help="Folder to save model weights")
parser.add_argument("--device", type=str, default="cuda:0", choices=["cpu", "cuda:0"], help="Device to use for training")
args = parser.parse_args()

# 加载数据集
train_dataset = DeepfakeDataset(data_folder=args.data_folder, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

# 权重文件路径
os.makedirs(args.weights_folder, exist_ok=True)  # 确保权重文件夹存在
model_weights_path = os.path.join(args.weights_folder, "deepfake_detection_model.pth")

# 获取模型
device = args.device
if os.path.exists(model_weights_path):
    model = get_model(weights_forgery_path=model_weights_path, device=device)
else:
    model = get_model(weights_forgery_path=None, device=device)

# 定义超参数
batch_size = args.batch_size
learning_rate = args.learning_rate
num_epochs = args.num_epochs

# 修改损失函数为 BCEWithLogitsLoss
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 使用 tqdm 显示训练进度
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    # 使用 tqdm 包装 train_loader
    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", unit="batch")
    for batch_idx, (data, labels, lengths) in enumerate(progress_bar):
        data, labels, lengths = data.to(device), labels.to(device), lengths.to(device)

        outputs = model(data, lengths)  # 模型输出为 logits
        loss = criterion(outputs.squeeze(), labels)  # BCEWithLogitsLoss 需要 logits 和 float 标签

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # 更新进度条信息
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

# 保存模型权重
torch.save({"model": model.state_dict()}, model_weights_path)
print(f"Face forgery weights saved at {model_weights_path}")