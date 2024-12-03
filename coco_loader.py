import torch
from torchvision import transforms
import numpy as np
import json
import os

class CustomCOCODataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, annotations_file, transform=None, batch_size=32):
        self.root_dir = root_dir
        self.annotations_file = annotations_file
        self.transform = transform
        self.batch_size = batch_size
        
        # Load the annotations and image paths
        with open(annotations_file) as f:
            self.annotations = json.load(f)
        
        # Load the preprocessed image batches (assuming batches of .npy files)
        self.image_batches = self._load_image_batches()

    def _load_image_batches(self):
        image_batches = []
        batch_files = [f for f in os.listdir(self.root_dir) if f.endswith('.npy')]
        batch_files.sort()  # Ensure batches are in correct order
        for batch_file in batch_files:
            # Using memory-mapped loading to avoid loading the entire array into memory
            batch_data = np.load(os.path.join(self.root_dir, batch_file), mmap_mode='r')
            image_batches.append(batch_data)
        return image_batches

    def __getitem__(self, index):
        # Calculate which batch and image to get based on index
        batch_index = index // self.batch_size  # Determine the batch
        image_batch = self.image_batches[batch_index]
        
        # Calculate the image index within the batch
        image_index = index % len(image_batch)
        
        image = image_batch[image_index]

        if self.transform:
            image = self.transform(image)
        
        label = self.annotations['annotations'][index]['category_id']  # Adjust based on your annotations format
        return image, label

    def __len__(self):
        return len(self.annotations['images'])  # Assuming annotations contains all images

# Transformation (for image preprocessing)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Example usage
root_dir = './preprocessed_data'  # Path to the preprocessed image batches
annotations_file = './preprocessed_data/annotations.json'  # Path to your annotations file

dataset = CustomCOCODataset(root_dir, annotations_file, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
