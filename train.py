import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast

# Path to preprocessed data
data_dir = 'preprocessed_data'  # This folder contains both images and captions

# Hyperparameters
batch_size = 16  # Adjust for RTX 3050
learning_rate = 1e-4
epochs = 10
embedding_dim = 256

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset class
class ImageCaptionDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = sorted(os.listdir(data_dir))
        self.image_files = [f for f in self.files if f.endswith('.npy')]
        self.caption_files = [f for f in self.files if f.endswith('.json')]
        assert len(self.image_files) == len(self.caption_files), "Mismatch between images and captions!"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load the image
        image_path = os.path.join(self.data_dir, self.image_files[idx])
        image = np.load(image_path, allow_pickle=True)
        if len(image.shape) == 3:  # (H, W, C)
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Convert to (C, H, W)
        elif len(image.shape) == 4:  # If it's already batched, take the first image
            image = torch.tensor(image[0], dtype=torch.float32).permute(2, 0, 1)
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        # Load the caption
        caption_path = os.path.join(self.data_dir, self.caption_files[idx])
        with open(caption_path, 'r') as f:
            caption_data = json.load(f)
        caption = caption_data[0] if caption_data else ""

        return image.to(device), caption


# Model class
class ImageCaptionModel(nn.Module):
    def __init__(self, embedding_dim):
        super(ImageCaptionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 112 * 112, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, 256)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Data loaders
def load_data():
    dataset = ImageCaptionDataset(data_dir)
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# Training function
def train_model():
    train_loader, val_loader = load_data()
    model = ImageCaptionModel(embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, captions) in enumerate(train_loader):
            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                dummy_target = torch.zeros(outputs.size(0), dtype=torch.long, device=device)
                loss = criterion(outputs, dummy_target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {running_loss / len(train_loader):.4f}")

        # Save checkpoint
        checkpoint_dir = "saved_models"
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth"))

# Entry point
if __name__ == "__main__":
    train_model()
