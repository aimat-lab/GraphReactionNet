import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
# TODO implement N1  and Huber Loss

class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper
    """loss ignore the outliers which is large than given threshold """
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, threshold=None, interpolate=True):
        
        mask = torch.abs(input - target) < threshold

        if mask is not None:
            input = input[mask]
            target = target[mask]
        mim = torch.min(torch.min(input), torch.min(target))
        g = torch.log(input+mim) - torch.log(target+mim)
        # n, c, h, w = g.shape
        # norm = 1/(h*w)
        # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        print(10 * torch.sqrt(Dg))
        return 10 * torch.sqrt(Dg)

class Custom_HuberLoss(nn.Module):
    def __init__(self):
        """adaptive huberloss for energy difference"""
        super(Custom_HuberLoss, self).__init__()
        self.name = 'Custom_HuberLoss'

    def forward(self, input, target, proportion=0.4, interpolate=True):
        
        delta = torch.abs(input - target) * proportion

        return torch.nn.HuberLoss(reduction='mean', delta=delta)

