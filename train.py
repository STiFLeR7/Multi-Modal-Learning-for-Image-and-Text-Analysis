import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset_loader import FlickrDataset
from model_architecture import CustomModel

class Config:
    image_dir = "D:/Flickr8k-Dataset/Augmented_Flickr8k"
    caption_file = "D:/Flickr8k-Dataset/Flickr8k_text/Flickr8k_augmented.token.txt"
    vocab_size = 5000
    embedding_dim = 20
    batch_size = 16
    num_epochs = 10
    learning_rate = 0.0001
    model_save_path = "custom_model_epoch_{epoch}.pth"  # Save model checkpoints
    best_model_path = "best_model.pth"  # Save the best model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_interval = 100  # Print progress every 'n' batches

def validate(model, criterion, val_loader):
    """Validate the model and calculate the average loss."""
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0

    with torch.no_grad():  # Disable gradient computation for validation
        for images, caption_tokens, targets in tqdm(val_loader, desc="Validating"):
            images = images.to(Config.device)
            caption_tokens = caption_tokens.to(Config.device)
            targets = targets.to(Config.device)

            outputs = model(images, caption_tokens)
            batch_size, seq_len, vocab_size = outputs.shape
            outputs = outputs.view(-1, vocab_size)
            targets = targets.view(-1)

            loss = criterion(outputs, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss

def train():
    # Initialize data transforms and dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = FlickrDataset(Config.image_dir, Config.caption_file, Config.vocab_size, transform=transform)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = CustomModel(embedding_dim=Config.embedding_dim, vocab_size=Config.vocab_size).to(Config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)

    # Variables for tracking the best model
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(Config.num_epochs):
        model.train()
        running_loss = 0.0

        print(f"Epoch {epoch + 1}/{Config.num_epochs} started...")
        for batch_idx, (images, caption_tokens, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.num_epochs}")):
            if images is None or caption_tokens is None or targets is None:
                print("Skipped batch due to missing data.")
                continue

            # Move data to the correct device
            images = images.to(Config.device)
            caption_tokens = caption_tokens.to(Config.device)
            targets = targets.to(Config.device)

            # Forward pass
            outputs = model(images, caption_tokens)
            batch_size, seq_len, vocab_size = outputs.shape
            outputs = outputs.view(-1, vocab_size)
            targets = targets.view(-1)

            # Compute loss
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Log progress after every `log_interval` batches
            if (batch_idx + 1) % Config.log_interval == 0:
                avg_loss = running_loss / (batch_idx + 1)
                print(f"Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {avg_loss:.4f}")

        # Compute epoch loss
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{Config.num_epochs}], Training Loss: {epoch_loss:.4f}")

        # Save checkpoint
        epoch_model_path = Config.model_save_path.format(epoch=epoch + 1)
        torch.save(model.state_dict(), epoch_model_path)
        print(f"Model checkpoint saved to {epoch_model_path}")

        # Validate the model
        val_loss = validate(model, criterion, val_loader)
        print(f"Validation Loss after Epoch {epoch + 1}: {val_loss:.4f}")

        # Update the best model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.best_model_path)
            print(f"Best model updated and saved to {Config.best_model_path}")

    print("Training completed.")

if __name__ == "__main__":
    train()
