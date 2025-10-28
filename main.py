import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    return parser.parse_args()

class simple_image_dataset(Dataset):
    def __init__(self, image_dir, keypoint_annotations=None, transform=None):
        self.image_dir = Path(image_dir)
        self.files = sorted([p for p in self.image_dir.iterdir() if p.suffix.lower() in {'.jpg','.png','.jpeg'}])
        self.annotations = keypoint_annotations
        self.transform = transform or transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        img = self.transform(img)
        keypoints = torch.zeros(17,2)
        return img, keypoints

class small_cnn_2d(nn.Module):
    def __init__(self, num_joints=17):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3,padding=1)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.fc1 = nn.Linear(128*32*32,512)
        self.fc2 = nn.Linear(512,num_joints*2)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x,2)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(x.size(0),-1,2)

class simple_temporal_lifter(nn.Module):
    def __init__(self, input_joints=17, input_dim=2, hidden=256, output_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_joints*input_dim,hidden)
        self.fc2 = nn.Linear(hidden,hidden)
        self.fc3 = nn.Linear(hidden,input_joints*output_dim)
    def forward(self, x):
        b = x.size(0)
        x = x.view(b,-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(b,-1,3)

class simple_lighting_estimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3,padding=1)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.fc1 = nn.Linear(64*32*32,128)
        self.fc2 = nn.Linear(128,4)
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def mpjpe(pred, target):
    return torch.mean(torch.norm(pred-target, dim=-1))

def train_epoch(models, optimizers, dataloader, device):
    detector, lifter, light = models
    optim_d, optim_lft, optim_light = optimizers
    detector.train()
    lifter.train()
    light.train()
    total_loss = 0.0
    for imgs, keypoints in dataloader:
        imgs = imgs.to(device)
        keypoints = keypoints.to(device)
        pred_2d = detector(imgs)
        loss_2d = F.mse_loss(pred_2d, keypoints)
        optim_d.zero_grad()
        loss_2d.backward()
        optim_d.step()
        pred_3d = lifter(pred_2d.detach())
        gt_3d = torch.zeros_like(pred_3d)
        loss_3d = F.mse_loss(pred_3d, gt_3d)
        optim_lft.zero_grad()
        loss_3d.backward()
        optim_lft.step()
        pred_light = light(imgs)
        gt_light = torch.zeros_like(pred_light)
        loss_light = F.mse_loss(pred_light, gt_light)
        optim_light.zero_grad()
        loss_light.backward()
        optim_light.step()
        total_loss += (loss_2d.item()+loss_3d.item()+loss_light.item())
    return total_loss/len(dataloader)

def evaluate(models, dataloader, device):
    detector, lifter, light = models
    detector.eval()
    lifter.eval()
    total_mpjpe = 0.0
    with torch.no_grad():
        for imgs, keypoints in dataloader:
            imgs = imgs.to(device)
            keypoints = keypoints.to(device)
            pred_2d = detector(imgs)
            pred_3d = lifter(pred_2d)
            gt_3d = torch.zeros_like(pred_3d)
            total_mpjpe += mpjpe(pred_3d, gt_3d).item()
    return total_mpjpe/len(dataloader)

def save_checkpoint(models, save_dir, epoch):
    os.makedirs(save_dir, exist_ok=True)
    for name, model in zip(['detector','lifter','light'], models):
        torch.save(model.state_dict(), os.path.join(save_dir, f'{name}_epoch_{epoch}.pt'))

def load_checkpoint(models, save_dir, epoch, device):
    for name, model in zip(['detector','lifter','light'], models):
        path = os.path.join(save_dir, f'{name}_epoch_{epoch}.pt')
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))

def make_dataloaders(data_root, batch_size):
    train_ds = simple_image_dataset(os.path.join(data_root,'train'))
    val_ds = simple_image_dataset(os.path.join(data_root,'val'))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader

def main():
    args = parse_args()
    device = args.device
    detector = small_cnn_2d().to(device)
    lifter = simple_temporal_lifter().to(device)
    light = simple_lighting_estimator().to(device)
    optim_d = torch.optim.Adam(detector.parameters(), lr=args.lr)
    optim_lft = torch.optim.Adam(lifter.parameters(), lr=args.lr)
    optim_light = torch.optim.Adam(light.parameters(), lr=args.lr)
    train_loader, val_loader = make_dataloaders(args.data_root, args.batch_size)
    for epoch in range(1, args.epochs+1):
        train_loss = train_epoch((detector,lifter,light), (optim_d,optim_lft,optim_light), train_loader, device)
        val_score = evaluate((detector,lifter,light), val_loader, device)
        save_checkpoint((detector,lifter,light), args.save_dir, epoch)
        print(f'epoch {epoch} train_loss {train_loss:.4f} val_mpjpe {val_score:.4f}')

if __name__ == '__main__':
    main()
