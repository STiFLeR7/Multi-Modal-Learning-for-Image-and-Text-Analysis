import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import json
import nltk
import random
from torchvision import transforms

nltk.download('punkt')

class CocoDataset(Dataset):
    def __init__(self, images_dir, captions_file, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        # Load the captions file
        with open(captions_file, 'r') as f:
            self.data = json.load(f)

        # Prepare the list of image ids and corresponding captions
        self.image_ids = []
        self.captions = []

        for item in self.data['images']:
            image_id = item['id']
            # Only process images that have associated captions
            captions_for_image = [caption['caption'] for caption in self.data['annotations'] if caption['image_id'] == image_id]
            if captions_for_image:
                self.image_ids.append(image_id)
                self.captions.append(captions_for_image)

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        image_path = os.path.join(self.images_dir, f'{image_id}.jpg')
        try:
            return Image.open(image_path)
        except FileNotFoundError:
            print(f"Warning: Image {image_path} does not exist.")
            return None  # or return a placeholder image

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        captions = self.captions[idx]

        image = self.load_image(image_id)
        if image is None:
            return None  # Skip if image is missing

        if self.transform:
            image = self.transform(image)

        # Randomly select a caption for the image
        caption = random.choice(captions)

        # Tokenize the caption
        caption_tokens = nltk.word_tokenize(caption.lower())

        return image, caption_tokens

# Parameters
images_dir = 'D:/COCO-DATASET/coco2017/images'  # Update with your images path
captions_file = 'D:/COCO-DATASET/coco2017/annotations/captions_train2017.json'  # Update with your annotations path

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset and DataLoader
dataset = CocoDataset(images_dir, captions_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Iterate through the dataset
for i, (images, captions) in enumerate(dataloader):
    if images is None:
        continue  # Skip the batch if any image is missing

    print(f"Batch {i} processed.")
    print(f"Image batch shape: {images.size()}")
    print(f"Captions: {captions}")
