import os
import random
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from dataset_loader import FlickrDataset
from model_architecture import CustomModel
from nltk.translate.bleu_score import sentence_bleu

# Configurations
class Config:
    image_dir = "D:/Flickr8k-Dataset/Flicker8k_Dataset"
    caption_file = "D:/Flickr8k-Dataset/Flickr8k_text/Flickr8k.token.txt"
    vocab_size = 5000
    embedding_dim = 20
    batch_size = 16
    model_path = "custom_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_ratio = [0.8, 0.1, 0.1]  # Training, Validation, Test splits

def augment_dataset():
    """Create an augmented version of the Flickr8k dataset."""
    augmentation_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        transforms.ToTensor(),
    ])

    print("Augmenting dataset with random transformations...")
    augmented_dataset = FlickrDataset(
        Config.image_dir,
        Config.caption_file,
        Config.vocab_size,
        transform=augmentation_transforms
    )
    print(f"Augmented dataset created with {len(augmented_dataset)} samples.")
    return augmented_dataset

def split_dataset():
    """Split the dataset into training, validation, and test sets."""
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = FlickrDataset(Config.image_dir, Config.caption_file, Config.vocab_size, transform=base_transform)
    
    total_len = len(dataset)
    train_len = int(Config.split_ratio[0] * total_len)
    val_len = int(Config.split_ratio[1] * total_len)
    test_len = total_len - train_len - val_len

    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])
    print(f"Dataset split into: Training ({len(train_set)}), Validation ({len(val_set)}), Test ({len(test_set)})")
    return train_set, val_set, test_set

def validate(model, dataloader, criterion):
    """Validate the model and calculate loss, accuracy, and BLEU score."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_bleu = 0.0

    with torch.no_grad():
        for batch_idx, (images, caption_tokens, targets) in enumerate(tqdm(dataloader, desc="Validating")):
            # Move data to the correct device
            images = images.to(Config.device)
            caption_tokens = caption_tokens.to(Config.device)
            targets = targets.to(Config.device)

            # Forward pass
            outputs = model(images, caption_tokens)  # [batch_size, seq_len, vocab_size]

            # Reshape outputs and targets for loss calculation
            batch_size, seq_len, vocab_size = outputs.shape
            outputs = outputs.view(-1, vocab_size)  # [batch_size * seq_len, vocab_size]
            targets = targets.view(-1)  # [batch_size * seq_len]

            # Calculate loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Debugging: Print outputs and targets for the first few batches
            if batch_idx < 5:
                print(f"Batch {batch_idx + 1}: Outputs shape: {outputs.shape}, Targets shape: {targets.shape}")
                print(f"Sample Targets: {targets[:10]}")
                print(f"Sample Outputs (top logits): {outputs[:10].argmax(dim=1)}")

            # Calculate accuracy
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == targets).sum().item()
            total_samples += targets.size(0)

            # Calculate BLEU score for each caption in the batch
            for i in range(batch_size):
                pred_tokens = predictions[i * seq_len:(i + 1) * seq_len].tolist()
                target_tokens = targets[i * seq_len:(i + 1) * seq_len].tolist()
                total_bleu += sentence_bleu([target_tokens], pred_tokens)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    avg_bleu = total_bleu / total_samples

    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation BLEU Score: {avg_bleu:.4f}")

if __name__ == "__main__":
    # Split dataset
    train_set, val_set, test_set = split_dataset()

    # Optionally augment the training set
    augmented_train_set = augment_dataset()

    # Validation DataLoader
    val_loader = DataLoader(val_set, batch_size=Config.batch_size, shuffle=False)

    # Load the model
    model = CustomModel(embedding_dim=Config.embedding_dim, vocab_size=Config.vocab_size)
    model.load_state_dict(torch.load(Config.model_path, map_location=Config.device))
    model.to(Config.device)

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Validate the model
    validate(model, val_loader, criterion)
