from transformers import BertTokenizer
from PIL import Image
from torchvision import transforms

# Initialize the tokenizer for text processing
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define the transformation pipeline (only basic transform, like resizing if needed)
image_transform = transforms.Compose([
    transforms.Resize((224, 224))  # Only resizing, no ToTensor() here
])

def preprocess_image(image_path):
    """
    Loads an image from a file path, converts it to RGB, and applies transformations.
    
    Parameters:
    - image_path (str): The path to the image file.
    
    Returns:
    - PIL Image: Preprocessed image ready for further transformation.
    """
    image = Image.open(image_path).convert("RGB")
    return image_transform(image)  # Return as PIL image, without ToTensor

def preprocess_text(caption):
    """
    Tokenizes and encodes the caption text for BERT input.
    
    Parameters:
    - caption (str): The text caption associated with an image.
    
    Returns:
    - input_ids (Tensor): Token IDs tensor.
    - attention_mask (Tensor): Attention mask tensor.
    """
    encoding = tokenizer(caption, padding="max_length", max_length=32, truncation=True, return_tensors="pt")
    return encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0)
