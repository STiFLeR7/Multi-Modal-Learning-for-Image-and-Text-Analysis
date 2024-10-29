from torch.utils.data import DataLoader
from model_architecture import MultiModalModel
from dataset import COCODataset
from data_preprocessing import image_transform
import torch
import labels

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(images, input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    print(f"Evaluation Accuracy: {accuracy * 100:.2f}%")

def main():
    dataset = COCODataset("val2017", "annotations/captions_val2017.json", transform=image_transform)
    dataloader = DataLoader(dataset, batch_size=16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MultiModalModel(num_classes=10).to(device)
    model.load_state_dict(torch.load("best_model.pth"))  # Load the trained model
    
    evaluate(model, dataloader, device)

if __name__ == "__main__":
    main()
