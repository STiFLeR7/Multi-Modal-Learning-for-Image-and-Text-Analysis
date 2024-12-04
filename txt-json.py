import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class CustomCOCODataset(Dataset):
    def __init__(self, root_dir, annotations_file, transform=None):
        self.root_dir = root_dir
        self.annotations = self.load_annotations(annotations_file)
        self.transform = transform
        self.batch_files = self._get_batch_files()

        print(f"Loaded {len(self.batch_files)} batch files")
        print(f"Loaded {len(self.annotations)} annotations")

    def load_annotations(self, annotations_file):
        # Load and validate the annotations from the JSON file
        with open(annotations_file, "r") as f:
            annotations = json.load(f)

        # Validation: Ensure it's a list of dictionaries
        if not isinstance(annotations, list):
            raise ValueError("Annotations must be a list")
        if not all(isinstance(item, dict) for item in annotations):
            raise ValueError("Each annotation must be a dictionary")

        # Debugging: Check the first few entries
        print(f"First 2 annotations: {annotations[:2]}")
        return annotations

    def _get_batch_files(self):
        # Collect all batch files from the directory
        batch_files = sorted(
            [
                os.path.join(self.root_dir, f)
                for f in os.listdir(self.root_dir)
                if f.startswith("images_batch_") and f.endswith(".npy")
            ]
        )
        return batch_files

    def __len__(self):
        # Return the total number of images (based on annotations)
        return len(self.annotations)

    def __getitem__(self, idx):
        try:
            # Get annotation
            annotation = self.annotations[idx]
            image_id = annotation["image_id"]
            predicted_class = annotation["predicted_class"]

            # Determine the batch file and index within the batch
            batch_idx = (image_id - 1) // 1000  # Assuming 1000 images per batch
            image_idx_within_batch = (image_id - 1) % 1000

            # Ensure the batch file exists
            if batch_idx >= len(self.batch_files):
                raise IndexError(f"Batch index {batch_idx} out of range for available batches")

            batch_file = self.batch_files[batch_idx]

            # Load the batch using memory mapping
            memmap = np.memmap(batch_file, dtype=np.float32, mode="r")
            total_pixels = memmap.shape[0]
            num_pixels_per_image = 224 * 224 * 3
            num_images = total_pixels // num_pixels_per_image

            # Ensure the memmap can be reshaped
            if total_pixels % num_pixels_per_image != 0:
                print(f"Warning: Incomplete batch detected in {batch_file}. Ignoring excess pixels.")
                num_images = total_pixels // num_pixels_per_image

            # Reshape the batch into (num_images, 224, 224, 3)
            batch_data = memmap[: num_images * num_pixels_per_image].reshape(
                (num_images, 224, 224, 3)
            )
            image = batch_data[image_idx_within_batch].astype(np.uint8)
            del memmap  # Free memory

            image = Image.fromarray(image)

            if self.transform:
                image = self.transform(image)

            return image, predicted_class
        except IndexError as e:
            print(f"IndexError at idx {idx}: {e}")
            raise e
        except ValueError as e:
            print(f"ValueError at idx {idx}: {e}")
            raise e


def run_inference_on_dataset(dataset_dir, model):
    # Assuming model is already loaded and ready for inference
    transform = None  # Add any transformations if needed
    dataset = CustomCOCODataset(
        root_dir=dataset_dir, annotations_file="./preprocessed_data/annotations.json", transform=transform
    )
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Run inference on the dataset
    for idx, (images, labels) in enumerate(data_loader):
        print(f"Processing batch {idx + 1}/{len(data_loader)}")
        # Perform inference with model on images here
        # Example: outputs = model(images)
        # Add saving results, logging, or other operations as needed


if __name__ == "__main__":
    dataset_dir = "./preprocessed_data"  # Path to your dataset directory
    model = "./coco_resnet18.pth"  # Load or initialize your model here
    run_inference_on_dataset(dataset_dir, model)
