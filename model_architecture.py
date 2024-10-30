import torch
import torch.nn as nn
import torchvision.models as models

class YourModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(YourModel, self).__init__()
        
        # Image feature extractor (using ResNet)
        self.resnet = models.resnet50(weights='DEFAULT')
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, hidden_size)
        
        # Text embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # LSTM for text processing
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        
        # Final linear layer for output
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
        
    def forward(self, images, captions):
        # Process images
        image_features = self.resnet(images)

        # Process captions
        embedded_captions = self.embedding(captions)
        lstm_out, _ = self.lstm(embedded_captions)
        
        # Get the last output of LSTM
        lstm_out = lstm_out[:, -1, :]
        
        # Concatenate features
        combined_features = torch.cat((image_features, lstm_out), dim=1)
        
        # Final output
        outputs = self.fc(combined_features)
        
        return outputs
