import torch
from torch.utils.data import DataLoader, random_split
from data.dataset import PoseDataset
from models.lifting_network import MartinezNet
from training.loss import mpjpe

def evaluate_model(data_path, checkpoint_path):
    dataset = PoseDataset(data_path)
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    
    # fixed seed so split matches training
    generator = torch.Generator().manual_seed(42)
    _, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    val_loader = DataLoader(val_set, batch_size=64)

    model = MartinezNet()
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            total_loss += mpjpe(outputs, targets).item()

    final_mpjpe = total_loss / len(val_loader)
    print(f'Final MPJPE: {final_mpjpe:.4f}m ({final_mpjpe*1000:.1f}mm)')

if __name__ == '__main__':
    evaluate_model('data/mpi_inf_3dhp_train.npz', 'checkpoints/best_model.pth')