import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, text_file, vocab_size, transform=None):
        self.image_dir = image_dir
        self.text_file = text_file
        self.vocab_size = vocab_size
        self.transform = transform

        self.data = []

        # Load image names and captions
        with open(text_file, 'r') as file:
            for line in file:
                # If the line contains captions, split by tab
                if '\t' in line:
                    image_name, caption = line.strip().split('\t')
                else:
                    image_name, caption = line.strip(), "Dummy caption"

                self.data.append((image_name, caption))

        print(f"Dataset loaded: {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, caption = self.data[idx]

        # Clean image_name by removing any trailing # and text
        image_name = image_name.split('#')[0]  # Take only the portion before the '#'

        # Construct the image path
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

        # Convert the caption to token IDs (dummy example)
        caption_tokens = torch.randint(0, self.vocab_size, (20,))  # Dummy tokens
        target = torch.zeros(self.vocab_size)  # Dummy target

        return image, caption_tokens, target
