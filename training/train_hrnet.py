import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from models.hrnet import HRNet
from data.coco_dataset import COCOKeypointDataset
import os

def heatmap_loss(predicted, target, visibility):
    num_joints = predicted.shape[1]
    loss = 0
    for j in range(num_joints):
        mask = visibility[:, j] > 0
        if mask.sum() == 0:
            continue
        loss += torch.mean((predicted[mask, j] - target[mask, j]) ** 2)
    return loss / num_joints

def train(img_dir, ann_file, epochs=20, batch_size=64, lr=1e-3):
    dataset = COCOKeypointDataset(img_dir=img_dir, ann_file=ann_file)
    
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = HRNet(num_keypoints=17).to(device)
    optimiser = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, heatmaps, visibility in train_loader:
            imgs = imgs.to(device)
            heatmaps = heatmaps.to(device)
            visibility = visibility.to(device)
            optimiser.zero_grad()
            outputs = model(imgs)
            loss = heatmap_loss(outputs, heatmaps, visibility)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, heatmaps, visibility in val_loader:
                imgs = imgs.to(device)
                heatmaps = heatmaps.to(device)
                visibility = visibility.to(device)
                outputs = model(imgs)
                val_loss += heatmap_loss(outputs, heatmaps, visibility).item()
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/hrnet_best.pth')
            print(f'  -> Saved best model')

if __name__ == '__main__':
    train(
        img_dir='/user/HS402/yv00051/com1027yv00051/FYP/image-to-pose/data/train2017',
        ann_file='/user/HS402/yv00051/com1027yv00051/FYP/image-to-pose/data/annotations/person_keypoints_train2017.json'
    )