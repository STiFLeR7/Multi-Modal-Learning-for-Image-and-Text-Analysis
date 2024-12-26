import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset_loader import FlickrDataset
from model_architecture import CustomModel

# Configurations
class Config:
    image_dir = "D:/Flickr8k-Dataset/Flicker8k_Dataset"
    caption_file = "D:/Flickr8k-Dataset/Flickr8k_text/Flickr8k.token.txt"  # Use a separate validation file if available
    vocab_size = 5000
    embedding_dim = 20
    batch_size = 16
    model_path = "custom_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Validation function
def validate():
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load validation dataset
    dataset = FlickrDataset(Config.image_dir, Config.caption_file, Config.vocab_size, transform=transform)
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=False)

    # Load the model
    model = CustomModel(embedding_dim=Config.embedding_dim, vocab_size=Config.vocab_size)
    model.load_state_dict(torch.load(Config.model_path, map_location=Config.device))
    model.to(Config.device)
    model.eval()

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    with torch.no_grad():
        for images, caption_tokens, targets in tqdm(dataloader, desc="Validating"):
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

            # Compute loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    # Calculate average loss
    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    validate()
