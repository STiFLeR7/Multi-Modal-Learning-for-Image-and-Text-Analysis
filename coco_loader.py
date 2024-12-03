import json
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class CocoDataset(Dataset):
    def __init__(self, annotation_file, root_dir, transform=None):
        self.annotation_file = annotation_file
        self.root_dir = root_dir
        self.transform = transform

        # Load the annotations JSON file
        with open(annotation_file, 'r') as file:
            self.annotations = json.load(file)
        
        # Validate structure
        if not isinstance(self.annotations, dict):
            raise ValueError(f"Expected dictionary in {annotation_file}, but found {type(self.annotations)}")
        
        self.images = self.annotations.get('images', [])
        self.annotations_data = self.annotations.get('annotations', [])
        
        if not self.images or not self.annotations_data:
            print(f"Warning: Missing 'images' or 'annotations' keys in {annotation_file}")

        print(f"Loaded {len(self.images)} images and {len(self.annotations_data)} annotations.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get the image file name and load the image
        image_info = self.images[idx]
        image_id = image_info.get('id')  # Get the image_id
        image_path = os.path.join(self.root_dir, image_info.get('file_name', ''))
        
        if not os.path.exists(image_path):
            print(f"Error: Image file {image_path} not found!")
            return None, None  # Return None if the image is missing

        image = Image.open(image_path).convert('RGB')

        # Retrieve annotations for the image
        annotations = [ann for ann in self.annotations_data if ann.get('image_id') == image_id]
        if not annotations:
            print(f"Warning: No annotations found for image_id {image_id}")

        captions = [ann['caption'] for ann in annotations if 'caption' in ann]

        if self.transform:
            image = self.transform(image)

        return image, captions
