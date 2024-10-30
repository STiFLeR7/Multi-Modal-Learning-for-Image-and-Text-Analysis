import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import COCODataset
from model_architecture import YourModel
from data_preprocessing import DataPreprocessor

# Set up device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 8
learning_rate = 0.001
num_epochs = 10
vocab_size = 30522  # Adjust based on your tokenizer
embed_size = 256
hidden_size = 512

# Load and preprocess data
image_dir = 'D:/COCO-DATASET/coco2017/train2017'
caption_file = 'D:/COCO-DATASET/coco2017/annotations/captions_train2017.json'

preprocessor = DataPreprocessor(image_dir, caption_file)
captions = preprocessor.load_data()

# Initialize dataset and dataloader
dataset = COCODataset(image_dir=image_dir, captions=captions, transform=preprocessor.transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
model = YourModel(vocab_size, embed_size, hidden_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(num_epochs):
    for images, captions in data_loader:
        images, captions = images.to(device), captions.to(device)

        # Forward pass
        outputs = model(images, captions)  # outputs: (batch_size, sequence_length, vocab_size)

        # Reshape outputs and captions for CrossEntropyLoss
        # outputs should be (batch_size * sequence_length, vocab_size)
        # captions should be (batch_size * sequence_length)
        outputs = outputs.view(-1, vocab_size)  # Reshape outputs
        captions = captions.view(-1)  # Reshape captions to match

        # Compute loss
        loss = criterion(outputs, captions)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete.")
