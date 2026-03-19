import torch
from torch.utils.data import DataLoader, Subset
from data.dataset import PoseDataset
from models.lifting_network import MartinezNet
from training.loss import mpjpe

# load only 1000 samples
dataset = PoseDataset('data/mpi_inf_3dhp_train.npz')
subset = Subset(dataset, range(1000))
loader = DataLoader(subset, batch_size=64, shuffle=True)

model = MartinezNet()
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    total_loss = 0
    for inputs, targets in loader:
        optimiser.zero_grad()
        outputs = model(inputs)
        loss = mpjpe(outputs, targets)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}')