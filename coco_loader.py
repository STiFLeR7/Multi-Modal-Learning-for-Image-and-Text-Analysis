import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json

class CustomCOCODataset(Dataset):
    def __init__(self, root_dir, annotations_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Load the annotations file (you need to ensure it's correctly formatted)
        self.annotations = self.load_annotations(annotations_file)
        
        # Initialize the list of image filenames (use .npy files)
        self.image_batches = self.load_image_batches()

    def load_annotations(self, file_path):
        # Load and return the annotations from the JSON file
        with open(file_path, 'r') as f:
            return json.load(f)

    def load_image_batches(self):
        # Here we assume the images are in .npy files
        image_batches = []
        for i in range(1, 6):  # Assuming you have 5 batches (adjust accordingly)
            image_batch = np.load(os.path.join(self.root_dir, f'images_batch_{i}.npy'))
            image_batches.append(image_batch)
        return image_batches

    def __len__(self):
        return sum([batch.shape[0] for batch in self.image_batches])  # Total number of images

    def __getitem__(self, idx):
        # Find the correct batch for the given index
        for batch in self.image_batches:
            if idx < batch.shape[0]:
                image = batch[idx]
                break
            else:
                idx -= batch.shape[0]
        
        # Convert numpy image to PIL for transformations
        image = Image.fromarray(image)
        
        # Apply transformations (resizing, normalization)
        if self.transform:
            image = self.transform(image)
        
        return image, 0  # 0 as a placeholder for the label (update this based on your annotation structure)

# Define transforms: Resize, ToTensor, Normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Example usage:
dataset = CustomCOCODataset(root_dir='./preprocessed_data', annotations_file='./preprocessed_data/annotations.json', transform=transform)
