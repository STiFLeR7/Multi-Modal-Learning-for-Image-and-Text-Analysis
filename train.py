import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset_loader import FlickrDataset
from model_architecture import CustomModel

# Configurations
class Config:
    image_dir = "D:/Flickr8k-Dataset/Flicker8k_Dataset"
    caption_file = "D:/Flickr8k-Dataset/Flickr8k_text/Flickr8k.token.txt"
    vocab_size = 5000
    embedding_dim = 20
    batch_size = 16
    num_epochs = 10
    learning_rate = 0.001
    model_save_path = "custom_model.pth"

# Training function
def train():
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Dataset and DataLoader
    dataset = FlickrDataset(Config.image_dir, Config.caption_file, Config.vocab_size, transform=transform)
    data_loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)

    # Model
    model = CustomModel(Config.embedding_dim, Config.vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)

    # Training loop
    for epoch in range(Config.num_epochs):
        model.train()
        running_loss = 0.0

        for images, texts, targets in tqdm(data_loader, desc=f"Epoch {epoch + 1}/{Config.num_epochs}"):
            # Forward pass
            outputs = model(images, texts)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(data_loader)
        print(f"Epoch [{epoch + 1}/{Config.num_epochs}], Loss: {avg_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), Config.model_save_path)
    print(f"Model saved to {Config.model_save_path}")

if __name__ == "__main__":
    train()
