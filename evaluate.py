import torch

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, captions in dataloader:
            images, captions = images.to(device), captions.to(device)
            outputs = model(images, captions)
            loss = criterion(outputs, captions.view(-1))
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Average Evaluation Loss: {avg_loss:.4f}")
    model.train()
    return avg_loss
