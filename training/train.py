import torch
from torch.utils.data import DataLoader, random_split
from data.dataset import PoseDataset
from models.lifting_network import MartinezNet
import matplotlib.pyplot as plt
from training.loss import mpjpe
import os


def train(data_path, epochs=100, batch_size=64, lr=1e-3, dropout=0.5):
    dataset = PoseDataset(data_path)
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    model = MartinezNet(dropout=dropout)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
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
                outputs = model(inputs)
                val_loss += mpjpe(outputs, targets).item()
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print(f'  -> Saved best model')

    # plot and save loss curve
    plt.figure()
    plt.plot(train_losses, label='Train MPJPE')
    plt.plot(val_losses, label='Val MPJPE')
    plt.xlabel('Epoch')
    plt.ylabel('MPJPE (m)')
    plt.title('Training Curve')
    plt.legend()
    plt.savefig('checkpoints/loss_curve.png')
    print('Loss curve saved to checkpoints/loss_curve.png')

if __name__ == '__main__':
    train('data/mpi_inf_3dhp_train.npz')