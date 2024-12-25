import torch
import torch.nn as nn
import torchvision.models as models

class CustomModel(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super(CustomModel, self).__init__()
        
        # Image Encoder: Use a pre-trained ResNet
        self.image_encoder = models.resnet18(pretrained=True)
        num_features = self.image_encoder.fc.in_features
        self.image_encoder.fc = nn.Linear(num_features, embedding_dim)
        
        # Text Encoder: Embedding layer for text
        self.text_encoder = nn.Embedding(vocab_size, embedding_dim)
        
        # Fusion Layer
        self.fusion_layer = nn.Linear(embedding_dim * 2, embedding_dim)
        
        # Output Layer
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, image, text):
        # Encode image: [batch_size, embedding_dim]
        image_features = self.image_encoder(image)
        
        # Encode text: [batch_size, seq_len, embedding_dim]
        text_features = self.text_encoder(text)
        
        # Repeat image features for each time step in the sequence
        image_features = image_features.unsqueeze(1).expand(-1, text_features.size(1), -1)
        
        # Concatenate image and text features: [batch_size, seq_len, embedding_dim * 2]
        combined_features = torch.cat((image_features, text_features), dim=2)
        
        # Fuse features: [batch_size, seq_len, embedding_dim]
        fused_features = self.fusion_layer(combined_features)
        
        # Generate output: [batch_size, seq_len, vocab_size]
        outputs = self.output_layer(fused_features)
        
        return outputs
