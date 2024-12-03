import torch
import torchvision.transforms as transforms
import torch.utils.data
from torchvision import models
import numpy as np
import os
import json
from PIL import Image

# Define the transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset class for COCO dataset
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
        
        # Check for correct structure of annotations
        # Adjust this line to match your annotations format
        annotation = self.annotations['annotations'][index]  # Get annotation for current index
        
        # Safely access the category_id key
        label = annotation.get('category_id', -1)  # Use .get() to handle missing keys safely
        
        return image, label

    def __len__(self):
        return len(self.annotations['images'])  # Assuming annotations contains all images

# Load the model
def load_model(model_path):
    model = models.resnet18(weights=None)  # Use appropriate weights parameter
    model.fc = torch.nn.Linear(model.fc.in_features, 80)  # Assuming there are 80 categories in the dataset
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    return model

# Function to run inference and save results
def run_inference_on_dataset(dataset_dir, model):
    dataset = CustomCOCODataset(dataset_dir, './preprocessed_data/annotations.json', transform=transform)  # Provide your annotations file
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)  # Disable multiprocessing
    
    # Create a directory to save the results
    results_dir = './inference_results'
    os.makedirs(results_dir, exist_ok=True)

    # Process the images
    processed_count = 0
    for idx, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Run the model on the images
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
        
        # Save the results (image id and predicted class)
        for i in range(images.size(0)):
            image_id = dataset.annotations['images'][idx * dataset.batch_size + i]['id']
            predicted_class = predicted[i].item()
            with open(os.path.join(results_dir, f'{image_id}_prediction.txt'), 'w') as f:
                f.write(f'Predicted Class: {predicted_class}\n')

        processed_count += len(images)
        print(f'Processed {processed_count}/{len(dataset)} images')

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Main block to prevent issues with multiprocessing on Windows
if __name__ == '__main__':
    # Load model
    model_path = './coco_resnet18.pth'  # Path to your model weights
    model = load_model(model_path).to(device)

    # Run inference on dataset
    dataset_dir = './preprocessed_data'  # Provide path to your preprocessed data
    run_inference_on_dataset(dataset_dir, model)
