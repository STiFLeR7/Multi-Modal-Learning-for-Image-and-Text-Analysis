import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
class FlickrDataset(Dataset):
    def __init__(self, image_dir, caption_file, vocab_size, transform=None):
        self.image_dir = image_dir
        self.vocab_size = vocab_size
        self.transform = transform or (lambda x: x)  # Default to identity transform if none provided
        self.skipped_files_log = "skipped_files.log"
        self.skipped_files = []

        # Load and preprocess captions
        self.data = []
        with open(caption_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    image_name = parts[0]
                    caption = parts[1]
                    image_path = os.path.join(image_dir, image_name.split("#")[0].strip())

                    # Check if the image file exists
                    if os.path.exists(image_path):
                        self.data.append((image_name, caption))
                    else:
                        self.skipped_files.append(image_path)

        # Log skipped files
        if self.skipped_files:
            with open(self.skipped_files_log, "w") as log_file:
                for file in self.skipped_files:
                    log_file.write(f"{file}\n")
            print(f"Logged {len(self.skipped_files)} skipped files to {self.skipped_files_log}")

        print(f"Dataset loaded with {len(self.data)} valid samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, caption = self.data[idx]

        # Clean image name
        image_name = image_name.split("#")[0].strip()
        image_path = os.path.join(self.image_dir, image_name)

        # Load and transform the image
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Tokenize the caption
        caption_tokens = self.convert_caption_to_tokens(caption)
        target = torch.tensor(caption_tokens, dtype=torch.long)

        return image, torch.tensor(caption_tokens, dtype=torch.long), target

    def convert_caption_to_tokens(self, caption):
        tokens = caption.split()[:20]  # Truncate to 20 tokens
        token_indices = [random.randint(0, self.vocab_size - 1) for _ in tokens]
        return token_indices + [0] * (20 - len(tokens))  # Pad to seq_len
