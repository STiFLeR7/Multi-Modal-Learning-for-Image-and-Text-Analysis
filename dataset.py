import json
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from data_preprocessing import preprocess_image, preprocess_text

# Define the complete transformation pipeline, including ToTensor and normalization
final_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class COCODataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        """
        COCO Dataset for image and caption pairs.
        
        Parameters:
        - image_dir (str): Path to the directory containing images.
        - annotation_file (str): Path to the JSON annotation file.
        - transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_dir = image_dir
        self.annotations = json.load(open(annotation_file, "r"))["annotations"]
        self.transform = transform or final_transform  # Use final_transform if no other transform is provided

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        # Construct the full path for the image
        image_path = os.path.join(self.image_dir, f"{annotation['image_id']:012d}.jpg")
        caption = annotation["caption"]
        
        # Preprocess image (returns PIL image)
        image = preprocess_image(image_path)
        
        # Apply the final transform to convert to tensor and normalize
        if self.transform:
            image = self.transform(image)
        
        # Preprocess text
        input_ids, attention_mask = preprocess_text(caption)
        
        return {
            "image": image,                  # Should be a tensor
            "input_ids": input_ids,          # Tensor from preprocess_text
            "attention_mask": attention_mask # Tensor from preprocess_text
        }

# Testing the Dataset class
if __name__ == "__main__":
    dataset = COCODataset(
        image_dir="D:/COCO-DATASET/coco2017/train2017",
        annotation_file="D:/COCO-DATASET/coco2017/annotations/captions_train2017.json"
    )
    
    # Print first sample to verify
    sample = dataset[0]
    print("Image Tensor Shape:", sample["image"].shape)
    print("Input IDs:", sample["input_ids"])
    print("Attention Mask:", sample["attention_mask"])
