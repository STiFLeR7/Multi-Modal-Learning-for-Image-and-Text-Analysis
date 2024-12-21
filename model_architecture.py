import torch
import torch.nn as nn
import torchvision.models as models

class CustomModel(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super(CustomModel, self).__init__()
        
        # Load ResNet18 with updated weights parameter
        from torchvision.models import ResNet18_Weights
        self.image_encoder = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify the fully connected layer
        num_features = self.image_encoder.fc.in_features
        self.image_encoder.fc = nn.Linear(num_features, embedding_dim)
        
        # Text encoder
        self.text_encoder = nn.Embedding(vocab_size, embedding_dim)
        
        # Fusion layer - Change output size to vocab_size
        self.fusion_layer = nn.Linear(embedding_dim * 2, vocab_size)  # Change to vocab_size

    def forward(self, image, text):
        # Encode image
        image_features = self.image_encoder(image)
        
        # Encode text
        text_features = self.text_encoder(text)  # [batch_size, seq_len, embedding_dim]
        text_features = text_features.mean(dim=1)  # Reduce seq_len dimension
        
        # Concatenate features
        combined_features = torch.cat((image_features, text_features), dim=1)
        
        # Fuse features
        output = self.fusion_layer(combined_features)
        
        return output

if __name__ == "__main__":
    # Example usage
    vocab_size = 5000
    embedding_dim = 20
    batch_size = 4
    
    # Initialize model
    model = CustomModel(embedding_dim, vocab_size)
    
    # Dummy input
    image_input = torch.randn(batch_size, 3, 224, 224)
    text_input = torch.randint(0, vocab_size, (batch_size, 20))
    
    # Forward pass
    output = model(image_input, text_input)
    print("Output shape:", output.shape)  # Should be [batch_size, vocab_size]