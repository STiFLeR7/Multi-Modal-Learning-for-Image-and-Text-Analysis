from PIL import Image
import os
import torch
from torch.utils.data import Dataset

class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, text_file, vocab_size, transform=None):
        self.image_dir = image_dir
        self.text_file = text_file
        self.vocab_size = vocab_size
        self.transform = transform

        # Load captions and image file names
        with open(text_file, 'r') as file:
            lines = file.readlines()

        self.data = []
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            image_name, caption = parts

            # Remove `#<number>` from image_name
            image_name = image_name.split('#')[0]
            
            self.data.append((image_name, caption))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, caption = self.data[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # Load and transform the image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}")

        if callable(self.transform):
            image = self.transform(image)
        else:
            raise TypeError("`self.transform` is not callable. Check its assignment in the dataset initialization.")

        # Convert caption to integers (dummy tokenization here)
        caption_tokens = torch.randint(0, self.vocab_size, (20,))  # Replace with actual tokenization logic
        target = torch.zeros(self.vocab_size)  # Dummy target for now

        return image, caption_tokens, target
