"""Frozen VGG19 feature extractor, retaining the original first 28 layers."""
import torch
from torchvision import models


class Vgg19(torch.nn.Module):
    def __init__(self, weights='imagenet'):
        super().__init__()
        weights = models.VGG19_Weights.IMAGENET1K_V1 if weights == 'imagenet' else None
        self.model = models.vgg19(weights=weights).features[:28]
        self.requires_grad_(False)
        self.eval()

    def forward(self, inputs):
        return self.model(inputs)
