import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset_loader import FlickrDataset
from model_architecture import CustomModel

# Configurations
class Config:
    image_dir = "D:/Flickr8k-Dataset/Flicker8k_Dataset"
    caption_file = "D:/Flickr8k-Dataset/Flickr8k_text/Flickr8k.token.txt"
    vocab_size = 5000
    embedding_dim = 20
    batch_size = 16
    num_epochs = 10
    learning_rate = 0.001
    model_save_path = "custom_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training function
def train():
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Dataset and DataLoader
    dataset = FlickrDataset(Config.image_dir, Config.caption_file, Config.vocab_size, transform=transform)
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)

    # Model, Loss, and Optimizer
    model = CustomModel(embedding_dim=Config.embedding_dim, vocab_size=Config.vocab_size).to(Config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)

    # Training Loop
    for epoch in range(Config.num_epochs):
        model.train()
        running_loss = 0.0

        for images, caption_tokens, targets in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{Config.num_epochs}"):
            # Move data to the correct device
            images = images.to(Config.device)
            caption_tokens = caption_tokens.to(Config.device)
            targets = targets.to(Config.device)

            # Forward pass
            outputs = model(images, caption_tokens)  # Output shape: [batch_size, seq_len, vocab_size]

            # Reshape outputs and targets for loss calculation
            batch_size, seq_len, vocab_size = outputs.shape
            outputs = outputs.view(-1, vocab_size)  # Reshape to [batch_size * seq_len, vocab_size]
            targets = targets.view(-1)  # Reshape to [batch_size * seq_len]

            # Calculate loss
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{Config.num_epochs}], Loss: {avg_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), Config.model_save_path)
    print(f"Model saved to {Config.model_save_path}")


if __name__ == "__main__":
    train()
