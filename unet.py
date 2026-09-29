import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tifffile
from sklearn.model_selection import train_test_split

# ==================== 1. 定义 U-Net 模型 ====================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bottom = DoubleConv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.outconv = nn.Conv2d(64, out_channels, 1)
        self.pool = nn.MaxPool2d(2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottom(self.pool(e4))
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.outconv(d1)
        return self.sigmoid(out)

# ==================== 2. 定义训练/验证数据集类（不变） ====================
class MembraneDataset(Dataset):
    def __init__(self, volume, labels, indices, transform=None):
        self.volume = volume
        self.labels = labels
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        image = self.volume[i].astype(np.float32) / 255.0   # 归一化到[0,1]
        mask = (self.labels[i] > 0).astype(np.float32)      # 二值化
        image = torch.from_numpy(image).unsqueeze(0)        # (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)          # (1, H, W)
        if self.transform:
            image = self.transform(image)
        return image, mask

# ==================== 2b. 定义仅用于测试的数据集类（新增，无标签） ====================
class TestDataset(Dataset):
    def __init__(self, volume, indices):
        self.volume = volume
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        image = self.volume[i].astype(np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)   # (1, H, W)
        return image

# ==================== 3. 加载训练和验证数据 ====================
print("加载训练和验证数据...")
volume = tifffile.imread(r"C:\Users\aa\Desktop\工作学习内容\现阶段任务\训练数据集\unet\train-volume.tif")
labels = tifffile.imread(r"C:\Users\aa\Desktop\工作学习内容\现阶段任务\训练数据集\unet\train-labels.tif")
indices = list(range(volume.shape[0]))
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
train_dataset = MembraneDataset(volume, labels, train_indices)
val_dataset = MembraneDataset(volume, labels, val_indices)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# ==================== 4. 初始化模型等 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=1, out_channels=1).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ==================== 5. 训练循环 ====================
num_epochs = 5
print(f"使用设备: {device}")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)
    val_loss /= len(val_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, 训练损失: {epoch_loss:.4f}, 验证损失: {val_loss:.4f}")

# ==================== 6. 测试阶段：仅预测并保存结果 ====================
print("加载测试图像...")
test_volume = tifffile.imread(r"C:\Users\aa\Desktop\工作学习内容\现阶段任务\训练数据集\unet\test-volume.tif")
test_indices = list(range(test_volume.shape[0]))
test_dataset = TestDataset(test_volume, test_indices)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model.eval()
with torch.no_grad():
    all_preds = []
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)                     # 输出概率图，形状 (1,1,H,W)
        pred = (outputs > 0.5).float()              # 阈值化
        pred = pred.squeeze().cpu().numpy()         # 转为 numpy 数组，形状 (H,W)
        all_preds.append(pred)

# 将预测结果堆叠为 (N, H, W) 并保存为 TIFF
all_preds = np.stack(all_preds, axis=0).astype(np.float32) * 255.0   # 转换回 0/255
tifffile.imwrite(r"C:\Users\aa\Desktop\工作学习内容\现阶段任务\训练数据集\unet\test-predictions.tif", all_preds)
print("预测结果已保存为 test-predictions.tif")

# ==================== 7. 保存模型 ====================
torch.save(model.state_dict(), "unet_membrane_segmentation.pth")
print("模型已保存为 unet_membrane_segmentation.pth")