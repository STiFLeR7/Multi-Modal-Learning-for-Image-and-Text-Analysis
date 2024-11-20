import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt

# Assuming the preprocessed data is in the 'preprocessed_data' folder
data_dir = 'preprocessed_data'  # Replace with your actual data directory

# Set embedding dimension (this can be tuned)
embedding_dim = 256  # Example embedding size for image features and text

# Define the ImageCaptionDataset class to load image and caption pairs
class ImageCaptionDataset(Dataset):
    def __init__(self, data_dir):
        """
        Custom Dataset to load images and captions from the same directory.
        Assumes that images are stored in .npy files and captions in .json files.
        """
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
        image = np.load(image_path, allow_pickle=True)  # Load image features
        
        # Ensure image shape is [batch_size, channels, height, width]
        image = torch.tensor(image, dtype=torch.float32)
        
        # Check the shape of the image, if it's [64, 3, 224, 224], we want to permute it
        if image.dim() == 4 and image.shape[1] == 64:
            image = image.permute(0, 2, 3, 1)  # Change the shape to [batch_size, height, width, channels]
        
        image = image.permute(0, 3, 1, 2)  # Now the shape is [batch_size, channels, height, width]

        # Load caption
        caption_path = os.path.join(self.data_dir, self.caption_files[idx])
        with open(caption_path, 'r') as f:
            caption_data = json.load(f)
        
        # Assuming each caption file contains a list of captions, use the first one
        caption = caption_data[0] if caption_data else ""  # Example: take the first caption
        return image, caption

# Define the ImageCaptionModel
class ImageCaptionModel(nn.Module):
    def __init__(self, embedding_dim):
        super(ImageCaptionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 112 * 112, embedding_dim)  # After pooling (224 -> 112)
        self.fc2 = nn.Linear(embedding_dim, 256)  # Adjust this based on your desired output dimension
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten the output for the FC layer
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = ImageCaptionModel(embedding_dim).to(device)
model.load_state_dict(torch.load('saved_models/model_epoch_10.pth'))  # Load the saved model weights
model.eval()  # Set the model to evaluation mode

# Load validation data
val_data = ImageCaptionDataset(data_dir=data_dir)  # Use the correct data directory
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

# Visualization function to display model predictions
def visualize_predictions(model, val_loader, device):
    model.eval()
    with torch.no_grad():
        for i, (images, captions) in enumerate(val_loader):
            if i >= 5:  # Show 5 images and captions for example
                break
            
            images = images.to(device)
            outputs = model(images)
            
            # Convert image tensor to numpy array for visualization
            img = images[0].cpu().numpy().transpose(1, 2, 0)
            img = np.clip(img, 0, 1)
            img = Image.fromarray((img * 255).astype(np.uint8))

            # Display the image and caption
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.title(f"Predicted Caption: {captions[0]}")
            plt.axis('off')
            plt.show()

# Example usage:
visualize_predictions(model, val_loader, device)
