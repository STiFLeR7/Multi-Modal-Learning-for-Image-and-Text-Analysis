import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Path to preprocessed data (both images and captions in this directory)
data_dir = 'preprocessed_data'  # Folder containing both images and captions

# Hyperparameters
batch_size = 32  # Reduced batch size to optimize memory usage
learning_rate = 1e-4
epochs = 10
embedding_dim = 256  # Example embedding size for image features and text

# Check if a GPU is available, and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the dataset class
class ImageCaptionDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = sorted(os.listdir(data_dir))  # List all files in the data directory

        # Separate the image and caption files
        self.image_files = [f for f in self.files if f.endswith('.npy')]
        self.caption_files = [f for f in self.files if f.endswith('.json')]

        # Ensure the number of image and caption files match
        assert len(self.image_files) == len(self.caption_files), "Mismatch between images and captions!"

        print(f"Loaded {len(self.image_files)} image files and {len(self.caption_files)} caption files.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.data_dir, self.image_files[idx])
        image = np.load(image_path, allow_pickle=True)
        image = torch.tensor(image, dtype=torch.float32)

        # Load caption
        caption_path = os.path.join(self.data_dir, self.caption_files[idx])
        with open(caption_path, 'r') as f:
            caption_data = json.load(f)

        caption = caption_data[0] if caption_data else ""  # Example: take the first caption
        return image, caption

# Define a simple neural network model (ImageCaptionModel)
class ImageCaptionModel(nn.Module):
    def __init__(self, embedding_dim):
        super(ImageCaptionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 112 * 112, embedding_dim)  # After pooling (224 -> 112)
        self.fc2 = nn.Linear(embedding_dim, 256)  # Output dimension, adjust as needed
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten the output for the FC layer
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load data
def load_data():
    dataset = ImageCaptionDataset(data_dir)
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
    # Use multiple workers to load data faster (parallel data loading)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Train Loader Size: {len(train_loader)} batches")
    print(f"Validation Loader Size: {len(val_loader)} batches")
    
    return train_loader, val_loader

# Define training function
def train_model():
    train_loader, val_loader = load_data()
    
    model = ImageCaptionModel(embedding_dim).to(device)  # Move model to GPU if available
    criterion = nn.CrossEntropyLoss()  # Example loss function for image-text matching
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Verify current working directory
    print(f"Current working directory: {os.getcwd()}")  # Debugging the directory issue

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        print(f"Epoch {epoch + 1}/{epochs} starting...")
        
        for batch_idx, (images, captions) in enumerate(train_loader):
            optimizer.zero_grad()

            # Move data to GPU if available
            images = images.to(device)

            # Check the batch size and image shape
            print(f"Batch {batch_idx + 1}/{len(train_loader)} - Image shape: {images.shape}")

            # Forward pass
            outputs = model(images)

            # Ensure captions are tensor with correct shape and format
            loss = criterion(outputs, torch.tensor([0]).to(device))  # Dummy loss (adjust as needed)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Debugging prints for loss and batch progress
            if batch_idx % 10 == 0:  # Reduced the frequency of print statements
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for images, captions in val_loader:
                images = images.to(device)
                outputs = model(images)
                loss = criterion(outputs, torch.tensor([0]).to(device))  # Dummy loss (adjust as needed)
                val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1}/{epochs}, Validation loss: {avg_val_loss:.4f}")

        # Save the model after each epoch
        model_save_path = f'model_epoch_{epoch + 1}.pth'
        try:
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved as {model_save_path}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")

# Main entry point
if __name__ == "__main__":
    train_model()
