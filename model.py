import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights  # Import ResNet18 weights enum
from pycocotools.coco import COCO
import os
import numpy as np
from tqdm import tqdm
from PIL import Image

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data Transformations for Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# COCO Dataset Path (adjust to your dataset path)
data_dir = "D:/COCO-DATASET/coco2017"
train_annotations = os.path.join(data_dir, 'annotations/instances_train2017.json')
train_images_dir = os.path.join(data_dir, 'train2017')

# Initialize COCO API for annotations
coco = COCO(train_annotations)

# Get category IDs
category_ids = coco.getCatIds()
categories = coco.loadCats(category_ids)
category_names = [category['name'] for category in categories]

# Create a dictionary mapping category name to an ID
category_to_id = {category['name']: category['id'] for category in categories}

# Dataset Class for COCO
class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, coco, transform=None, images_dir=None, num_classes=80):
        self.coco = coco
        self.transform = transform
        self.images_dir = images_dir
        self.num_classes = num_classes
        self.image_ids = coco.getImgIds()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # Get image and annotations
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.images_dir, img_info['file_name'])

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Load annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        # Get the category IDs from annotations (multi-label)
        labels = [ann['category_id'] for ann in annotations]

        # Create a one-hot encoded label vector
        one_hot_labels = torch.zeros(self.num_classes)
        for label in labels:
            if label - 1 < self.num_classes:  # Ensure the label is within bounds
                one_hot_labels[label - 1] = 1  # Subtract 1 to match the 0-based index in PyTorch

        if self.transform:
            image = self.transform(image)

        return image, one_hot_labels


def main():
    # Initialize Dataset and DataLoader
    train_dataset = CocoDataset(coco, transform=transform, images_dir=train_images_dir, num_classes=80)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # Define Model
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # Use the correct weights parameter
    model.fc = nn.Linear(model.fc.in_features, 80)  # Adjust the output layer for the number of classes
    model = model.to(device)

    # Loss Function and Optimizer
    criterion = nn.BCEWithLogitsLoss()  # For multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training Loop
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy (for multi-label)
            preds = torch.sigmoid(outputs) > 0.5  # Threshold at 0.5
            correct_preds += torch.sum(preds == labels).item()
            total_preds += labels.size(0) * labels.size(1)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_preds / total_preds

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "coco_resnet18.pth")

if __name__ == '__main__':
    main()
