import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import COCODataset, collate_fn
import torchvision.models as models

# Model class
class YourModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(YourModel, self).__init__()
        
        # Image feature extractor
        self.resnet = models.resnet50(weights='DEFAULT')
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, hidden_size)
        
        # Text embedding and LSTM
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        
        # Final output layer
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, images, captions):
        image_features = self.resnet(images)  # Shape: (batch_size, hidden_size)
        embedded_captions = self.embedding(captions)  # Shape: (batch_size, seq_len, embed_size)
        lstm_out, _ = self.lstm(embedded_captions)  # Shape: (batch_size, seq_len, hidden_size)
        lstm_out = lstm_out[:, -1, :]  # Get the last LSTM output for each sequence
        
        combined_features = torch.cat((image_features, lstm_out), dim=1)  # Shape: (batch_size, hidden_size * 2)
        outputs = self.fc(combined_features)  # Shape: (batch_size, vocab_size)
        return outputs

# Parameters
batch_size = 8
learning_rate = 0.001
num_epochs = 10
vocab_size = 30522  # Adjust based on your tokenizer's vocab size
embed_size = 256
hidden_size = 512

# Initialize dataset and dataloader
image_dir = "D:/COCO-DATASET/coco2017/train2017"
caption_file = "D:/COCO-DATASET/coco2017/annotations/captions_train2017.json"

dataset = COCODataset(image_dir=image_dir, caption_file=caption_file, transform=None)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Model, loss function, and optimizer
model = YourModel(vocab_size, embed_size, hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.train()

for epoch in range(num_epochs):
    for images, captions in data_loader:
        images, captions = images.to(device), captions.to(device)

        # Forward pass
        outputs = model(images, captions)  # Shape: (batch_size, vocab_size)

        # Flatten outputs and captions for the criterion
        loss = criterion(outputs.view(-1, vocab_size), captions[:, -1])  # Taking only last token for simplicity

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training complete.")
