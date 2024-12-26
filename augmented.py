import os
import random
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

class Config:
    original_image_dir = "D:/Flickr8k-Dataset/Flicker8k_Dataset"
    original_caption_file = "D:/Flickr8k-Dataset/Flickr8k_text/Flickr8k.token.txt"
    augmented_image_dir = "D:/Flickr8k-Dataset/Augmented_Flickr8k"
    augmented_caption_file = "D:/Flickr8k-Dataset/Flickr8k_text/Flickr8k_augmented.token.txt"
    num_augmentations = 3  # Number of augmented images per original image

def augment_image(image):
    augmentation_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
    ])
    return augmentation_transforms(image)

def create_augmented_dataset():
    os.makedirs(Config.augmented_image_dir, exist_ok=True)

    with open(Config.original_caption_file, "r") as file:
        captions = file.readlines()

    augmented_captions = []
    for line in tqdm(captions, desc="Augmenting dataset"):
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        original_image_name, caption = parts
        original_image_path = os.path.join(Config.original_image_dir, original_image_name.split("#")[0])

        if not os.path.exists(original_image_path):
            continue

        original_image = Image.open(original_image_path).convert("RGB")
        augmented_captions.append(f"{original_image_name}\t{caption}")

        for i in range(Config.num_augmentations):
            augmented_image = augment_image(original_image)
            augmented_image_name = f"{original_image_name.split('.')[0]}_aug{i}.jpg"
            augmented_image_path = os.path.join(Config.augmented_image_dir, augmented_image_name)
            augmented_image.save(augmented_image_path)
            augmented_captions.append(f"{augmented_image_name}\t{caption}")

    with open(Config.augmented_caption_file, "w") as file:
        file.writelines("\n".join(augmented_captions))
    print("Augmented dataset created.")

if __name__ == "__main__":
    create_augmented_dataset()
