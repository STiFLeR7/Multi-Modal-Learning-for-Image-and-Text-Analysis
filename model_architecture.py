import torch
import torch.nn as nn
from transformers import BertModel

class MultiModalModel(nn.Module):
    def __init__(self, num_classes, text_embedding_dim=768, image_embedding_dim=512, fusion_dim=1024):
        super(MultiModalModel, self).__init__()
        
        # Text encoder (BERT)
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.text_fc = nn.Linear(text_embedding_dim, fusion_dim)
        
        # Image encoder (CNN - e.g., ResNet)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, image_embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.image_fc = nn.Linear(image_embedding_dim, fusion_dim)
        
        # Fusion layer
        self.fc = nn.Linear(fusion_dim * 2, num_classes)
        
    def forward(self, image, text_input_ids, text_attention_mask):
        # Text encoding
        text_features = self.text_encoder(text_input_ids, attention_mask=text_attention_mask).last_hidden_state[:, 0, :]
        text_features = self.text_fc(text_features)
        
        # Image encoding
        image_features = self.image_encoder(image).squeeze()
        image_features = self.image_fc(image_features)
        
        # Fusion
        fused_features = torch.cat((text_features, image_features), dim=1)
        output = self.fc(fused_features)
        
        return output
