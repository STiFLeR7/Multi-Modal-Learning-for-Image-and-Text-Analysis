import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the dataset class for loading preprocessed images
class CustomCOCODataset(Dataset):
    def __init__(self, root_dir, annotations_file, transform=None):
        self.root_dir = root_dir
        self.annotations = self.load_annotations(annotations_file)
        self.transform = transform
        self.image_batches = self._load_image_batches()

        # Debugging: Print the number of batches and their shapes
        print(f"Loaded {len(self.image_batches)} batches")
        for i, batch in enumerate(self.image_batches):
            print(f"Batch {i} shape: {batch.shape}")

    def load_annotations(self, annotations_file):
        # Load and return the annotations from the JSON file
        with open(annotations_file) as f:
            return json.load(f)

    def _load_image_batches(self):
        # Load the preprocessed image batches using memory mapping
        image_batches = []
        for batch_file in os.listdir(self.root_dir):
            if batch_file.endswith('.npy'):
                # Use memory-mapping to load the array without using too much memory
                batch_data = np.memmap(os.path.join(self.root_dir, batch_file), dtype='float32', mode='r')
                image_batches.append(batch_data)
        return image_batches

    def __len__(self):
        # Return the total number of images
        return len(self.annotations['images'])

    def __getitem__(self, idx):
        # Debugging: Ensure the index is within bounds
        try:
            image_id = self.annotations['images'][idx]['id']
            image_batch_idx = idx // 1000  # Divide by batch size to get the batch index
            image_id_within_batch = idx % 1000  # Get the index within the batch

            # Ensure that the batch exists and has enough images
            image_batch = self.image_batches[image_batch_idx]
            print(f"Accessing batch {image_batch_idx}, image {image_id_within_batch}")
            if image_batch.shape[0] <= image_id_within_batch:
                raise IndexError(f"Index {image_id_within_batch} is out of bounds for batch {image_batch_idx}. Batch size: {image_batch.shape[0]}")

            # Calculate number of images in the batch
            num_pixels_per_image = 224 * 224 * 3  # For 224x224 RGB images
            num_images = image_batch.shape[0] // num_pixels_per_image

            if image_batch.shape[0] % num_pixels_per_image != 0:
                raise ValueError(f"Cannot reshape array of size {image_batch.shape[0]} into image batches of size {num_pixels_per_image}.")

            # Reshape the batch data into (num_images, 224, 224, 3)
            image_batch = image_batch.reshape(num_images, 224, 224, 3)
            image = image_batch[image_id_within_batch]

            image = Image.fromarray(image.astype(np.uint8))  # Convert to uint8 for PIL

            if self.transform:
                image = self.transform(image)

            # Get the label
            label = self.annotations['annotations'][idx]['category_id']

            return image, label
        except IndexError as e:
            print(f"IndexError at idx {idx}: {e}")
            raise e
        except ValueError as e:
            print(f"ValueError at idx {idx}: {e}")
            raise e


# Load the pre-trained ResNet18 model
def load_model(model_path):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 80)  # Adjust for the number of classes (80 categories)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Function to run inference and save results in JSON format
def run_inference_on_dataset(dataset_dir, model):
    dataset = CustomCOCODataset(dataset_dir, './preprocessed_data/annotations.json', transform=transform)  # Provide your annotations file
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)  # Disable multiprocessing
    
    # Create a directory to save the results
    results_dir = './inference_results'
    os.makedirs(results_dir, exist_ok=True)

    # Dictionary to store the results
    results = {}

    # Process the images
    processed_count = 0
    for idx, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Run the model on the images
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
        
        # Store the predictions in the results dictionary
        for i in range(images.size(0)):
            image_id = dataset.annotations['images'][idx * dataset.batch_size + i]['id']
            predicted_class = predicted[i].item()
            results[image_id] = predicted_class

        processed_count += len(images)
        print(f'Processed {processed_count}/{len(dataset)} images')

    # Save the results as a JSON file
    with open(os.path.join(results_dir, 'predictions.json'), 'w') as f:
        json.dump(results, f, indent=4)

# Transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Main block to prevent issues with multiprocessing on Windows
if __name__ == '__main__':
    # Load model
    model_path = './coco_resnet18.pth'  # Path to your model weights
    model = load_model(model_path).to(device)

    # Run inference on dataset
    dataset_dir = './preprocessed_data'  # Provide path to your preprocessed data
    run_inference_on_dataset(dataset_dir, model)
