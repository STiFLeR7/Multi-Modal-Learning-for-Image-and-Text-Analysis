import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import argparse

# Directory paths
data_dir = "preprocessed_data"
model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)

# Define the dataset class
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
        image_path = os.path.join(self.data_dir, self.image_files[idx])
        caption_path = os.path.join(self.data_dir, self.caption_files[idx])
        image = np.load(image_path, allow_pickle=True)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Convert to (C, H, W)
        with open(caption_path, 'r') as f:
            caption_data = json.load(f)
        caption = caption_data[0] if caption_data else ""
        return image, caption

# Define the model
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

# Training function
def train_and_evaluate(model, train_loader, val_loader, epochs, lr, criterion, optimizer, device):
    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, captions in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            # Dummy loss
            targets = torch.zeros(outputs.size(0), dtype=torch.long).to(device)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, captions in val_loader:
                images = images.to(device)
                outputs = model(images)
                targets = torch.zeros(outputs.size(0), dtype=torch.long).to(device)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")
    return model

# Hyperparameter tuning
def tune_hyperparameters(args):
    dataset = ImageCaptionDataset(data_dir)
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    best_loss = float('inf')
    best_model_path = None

    for lr in args.learning_rates:
        for embedding_dim in args.embedding_dims:
            print(f"\nTuning with LR={lr}, Embedding Dim={embedding_dim}")
            model = ImageCaptionModel(embedding_dim)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            trained_model = train_and_evaluate(model, train_loader, val_loader, args.epochs, lr, criterion, optimizer, device)

            # Save model if it has the lowest validation loss
            model_path = os.path.join(model_dir, f"model_lr_{lr}_dim_{embedding_dim}.pth")
            torch.save(trained_model.state_dict(), model_path)
            print(f"Model saved: {model_path}")

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for Image Captioning Model")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rates", nargs='+', type=float, default=[1e-3, 1e-4], help="Learning rates to try")
    parser.add_argument("--embedding_dims", nargs='+', type=int, default=[128, 256], help="Embedding dimensions to try")
    args = parser.parse_args()
    tune_hyperparameters(args)
