import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import COCODataset, collate_fn  # Import collate_fn from dataset.py
import torchvision.models as models

# Define the model class directly in train.py
class YourModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(YourModel, self).__init__()
        
        # Image feature extractor (using ResNet for example)
        self.resnet = models.resnet50(weights='DEFAULT')  # Use `weights='DEFAULT'` for latest version
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, hidden_size)  # Replace final layer
        
        # Text embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # LSTM for text processing
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        
        # Final linear layer for output
        self.fc = nn.Linear(hidden_size * 2, vocab_size)  # Concatenated image and text features
        
    def forward(self, images, captions):
        # Process images
        image_features = self.resnet(images)

        # Process captions
        embedded_captions = self.embedding(captions)
        lstm_out, _ = self.lstm(embedded_captions)
        
        # Get the last output of LSTM
        lstm_out = lstm_out[:, -1, :]  # Take the last time step

        # Concatenate features
        combined_features = torch.cat((image_features, lstm_out), dim=1)
        
        # Final output
        outputs = self.fc(combined_features)
        
        return outputs

# Hyperparameters
batch_size = 8
learning_rate = 0.001
num_epochs = 10
vocab_size = 30522  # Adjust based on your dataset vocabulary size (e.g., BERT vocab size)
embed_size = 256
hidden_size = 512

# Load dataset and DataLoader with custom collate_fn
image_dir = 'D:/COCO-DATASET/coco2017/train2017'  # Update with your image directory
caption_file = 'D:/COCO-DATASET/coco2017/annotations/captions_train2017.json'  # Update with your caption file path
dataset = COCODataset(image_dir=image_dir, annotations_file=caption_file, transform=None)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Initialize model, loss function, and optimizer
model = YourModel(vocab_size, embed_size, hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Check if a GPU is available and move the model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
model.train()
for epoch in range(num_epochs):
    for images, captions in data_loader:
        # Move images and captions to device
        images = images.to(device)
        captions = captions.to(device)

        # Forward pass
        outputs = model(images, captions)
        
        # Compute loss
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))  # Reshape for correct batch and sequence

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete.")
