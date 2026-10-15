"""CrossEntropyLoss2D extracted from the original Loss_all.py."""
from torch import nn
from torch.nn import functional as F


class CrossEntropyLoss2D(nn.Module):
    def forward(self, inputs, target):
        if inputs.shape[-2:] != target.shape[-2:]:
            inputs = F.interpolate(inputs, size=target.shape[-2:], mode='bilinear', align_corners=True)
        return F.cross_entropy(inputs, target, ignore_index=250)
