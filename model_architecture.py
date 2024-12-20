import torch
import torch.nn as nn
import torchvision.models as models

class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(ImageCaptioningModel, self).__init__()

        # Pretrained CNN for feature extraction
        resnet = models.resnet18(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)

        # LSTM for caption generation
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, images, captions):
        # Extract features from images
        with torch.no_grad():
            features = self.feature_extractor(images)
        features = features.view(features.size(0), -1)
        features = self.fc(features)

        # Embed captions and concatenate with image features
        embeddings = self.embed(captions[:, :-1])  # Exclude <end> token during training
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)

        # Pass through LSTM and decode to vocabulary
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)

        return outputs

if __name__ == "__main__":
    # Example usage
    BATCH_SIZE = 4
    EMBED_SIZE = 256
    HIDDEN_SIZE = 512
    VOCAB_SIZE = 5000
    SEQ_LENGTH = 20

    model = ImageCaptioningModel(EMBED_SIZE, HIDDEN_SIZE, VOCAB_SIZE)

    # Dummy data
    images = torch.randn(BATCH_SIZE, 3, 224, 224)  # Batch of images
    captions = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH))  # Random captions

    outputs = model(images, captions)
    print("Output shape:", outputs.shape)  # Should be (BATCH_SIZE, SEQ_LENGTH - 1, VOCAB_SIZE)
