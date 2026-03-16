import torch
import torch.optim as optim
from models.lifting_network import MartinezNet
from training.loss import mpjpe

def train_one_epoch(model, loader, optimiser):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        optimiser.zero_grad()
        outputs = model(inputs)
        loss = mpjpe(outputs, targets)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            loss = mpjpe(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(loader)