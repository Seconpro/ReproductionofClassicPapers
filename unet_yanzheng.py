import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import tifffile
from sklearn.model_selection import train_test_split

# 复用原来的模型定义（从 unet.py 导入或复制过来）
from unet import UNet, MembraneDataset  # 假设 unet.py 中定义了这些类

# 加载数据并划分（与训练时一致）
volume = tifffile.imread(r"C:\Users\aa\Desktop\工作学习内容\现阶段任务\训练数据集\unet\train-volume.tif")
labels = tifffile.imread(r"C:\Users\aa\Desktop\工作学习内容\现阶段任务\训练数据集\unet\train-labels.tif")
indices = list(range(volume.shape[0]))
_, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

val_dataset = MembraneDataset(volume, labels, val_indices)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load("unet_membrane_segmentation.pth"))
model.eval()

def dice_coefficient(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()

dice_scores = []
with torch.no_grad():
    for images, masks in val_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        dice = dice_coefficient(outputs, masks)
        dice_scores.append(dice)

print(f"验证集平均 Dice 系数: {np.mean(dice_scores):.4f}")