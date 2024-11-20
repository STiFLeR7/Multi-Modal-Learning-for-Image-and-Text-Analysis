import torch
from vit_model import VisionTransformer

def main():
    # Initialize Vision Transformer
    model = VisionTransformer(pretrained=False)
    model.eval()  # Set to evaluation mode

    # Test with a dummy image
    dummy_image = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 channels, 224x224 image
    with torch.no_grad():
        output = model(dummy_image)
    print("Dummy Image Output Shape:", output.shape)

    # Test with a real image (optional)
    # Uncomment the following lines to test with a real image
    # from PIL import Image
    # from torchvision import transforms
    # image_path = "path/to/your/image.jpg"  # Update the path to an image
    # image = Image.open(image_path).convert("RGB")

    # # Transform the image
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    # image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # with torch.no_grad():
    #     output = model(image_tensor)
    # print("Real Image Output Shape:", output.shape)

if __name__ == "__main__":
    main()
