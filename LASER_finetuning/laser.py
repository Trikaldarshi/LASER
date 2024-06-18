"""

LASER Finetuning for HuBERT
Contrastive-IDM Loss used in the paper
Author: Amit Meghanani

Contact: ameghanani1@sheffield.ac.uk

"""

import torch
import torch.nn.functional as F

class IDMContrastiveLoss(torch.nn.Module):
    def __init__(self, sigma, margin):
        super(IDMContrastiveLoss, self).__init__()
        self.sigma = sigma
        self.margin = margin
        
    def forward(self, embeddings):
        # Compute pairwise Euclidean distances
        D_X = torch.cdist(embeddings, embeddings, p=2)
        D_X = D_X.squeeze(0)
        
        n = D_X.size(0)
        
        # Create mask tensor y using torch.where
        idx = torch.arange(n,device=D_X.device).view(n, 1)
        y = torch.where(torch.abs(idx - idx.T) <= self.sigma, torch.tensor(0, device=D_X.device), torch.tensor(1,device=D_X.device))
        
        # Calculate W_bar and W using broadcasting
        i, j = torch.meshgrid(torch.arange(n,device=D_X.device), torch.arange(n,device=D_X.device))
        W_bar = (i - j) ** 2 + 1
        W = 1 / W_bar

        # Compute the loss using element-wise operations
        loss = torch.sum(y * W_bar * F.relu(self.margin - D_X) + (1 - y) * W * D_X)
        # print which variable is on which device used in the loss



        # Normalize the loss by the number of elements in D_X
        loss = loss / (n * n)
        
        return loss

