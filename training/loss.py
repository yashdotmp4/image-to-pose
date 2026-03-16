import torch

def mpjpe(predicted, target):
    # predicted, target shape: (batch, 16, 3)
    return torch.mean(torch.norm(predicted - target, dim=-1))