import torch
from torch import nn as nn
from torch.nn import functional as F

class EdgeLoss(nn.Module):
    """Edge Loss using Sobel operator"""

    def __init__(self, loss_type='l1', reduction='mean', loss_weight=1.0, channels=4):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.channels = channels

        kernel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)
        kernel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32)

        self.register_buffer('kernel_x', kernel_x.view(1,1,3,3).repeat(channels,1,1,1))
        self.register_buffer('kernel_y',kernel_y.view(1,1,3,3).repeat(channels,1,1,1))

    def forward(self, pred, target):
        gx_pred = F.conv2d(pred, self.kernel_x, padding=1, groups=self.channels)
        gy_pred = F.conv2d(pred, self.kernel_y, padding=1, groups=self.channels)
        gx_target = F.conv2d(target, self.kernel_x, padding=1, groups=self.channels)
        gy_target = F.conv2d(target, self.kernel_y, padding=1, groups=self.channels)

        if self.loss_type == 'l1':
            loss = F.l1_loss(gx_pred, gx_target, reduction = self.reduction) + \
                    F.l1_loss(gy_pred, gy_target, reduction=self.reduction)
        else:
            loss = F.mse_loss(gx_pred, gx_target, reduction = self.reduction) + \
                    F.mse_loss(gy_pred, gy_target, reduction=self.reduction)

        return self.loss_weight * loss



