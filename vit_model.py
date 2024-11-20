import torch
import torch.nn as nn
from torchvision import models

class VisionTransformer(nn.Module):
    def __init__(self, pretrained=False, num_classes=80):
        super(VisionTransformer, self).__init__()
        
        # Load the pre-trained ViT model (using weights instead of the deprecated 'pretrained' parameter)
        self.vit = models.vit_b_16(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Access the final layer (classifier) and modify it for COCO (80 classes)
        # The classifier layer is stored as `self.vit.heads`, which is a `Sequential` object.
        self.vit.heads = nn.Sequential(
            nn.LayerNorm(self.vit.heads[0].in_features),
            nn.Linear(self.vit.heads[0].in_features, num_classes)
        )
        
    def forward(self, x):
        return self.vit(x)
