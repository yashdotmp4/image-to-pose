import torch
from torch.utils.data import DataLoader, random_split
from data.dataset import PoseDataset
from models.lifting_network import MartinezNet
import os

def mpjpe(predicted, target):
    return torch.mean(torch.norm(predicted - target, dim=-1))

def train(data_path, epochs=200, batch_size=32, lr=1e-3, dropout=0.5):
    dataset = PoseDataset(data_path)
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, drop_last=True)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = MartinezNet(num_joints_in=17, num_joints_out=17, dropout=dropout).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimiser.zero_grad()
            outputs = model(inputs)
            loss = mpjpe(outputs, targets)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += mpjpe(outputs, targets).item()
        val_loss /= len(val_loader)

        print(f'Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print(f'  -> Saved best model')

if __name__ == '__main__':
    train('/scratch/fyp-stuff/3dpw_hrnet_full.npz', epochs=200)