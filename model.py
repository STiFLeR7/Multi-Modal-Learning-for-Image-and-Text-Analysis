import torch
import torch.nn as nn
import torchvision.models as models

class CustomImageCaptioningModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers):
        super(CustomImageCaptioningModel, self).__init__()

        # Pretrained CNN as feature extractor (ResNet18 for simplicity)
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, embedding_dim)

        # Recurrent neural network for caption generation
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Fully connected layer to map RNN outputs to vocabulary size
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, captions):
        """
        Args:
            images: Batch of images (tensor of shape [batch_size, 3, 224, 224]).
            captions: Padded sequence of captions (tensor of shape [batch_size, max_seq_len]).

        Returns:
            outputs: Predicted logits for each word in the captions.
        """
        # Extract image features
        img_features = self.cnn(images)  # Shape: [batch_size, embedding_dim]

        # Prepare inputs for RNN
        img_features = img_features.unsqueeze(1)  # Add sequence dimension: [batch_size, 1, embedding_dim]
        rnn_inputs = torch.cat((img_features, captions[:, :-1]), dim=1)  # Shape: [batch_size, seq_len, embedding_dim]

        # Generate captions with RNN
        rnn_outputs, _ = self.rnn(rnn_inputs)

        # Map RNN outputs to vocabulary space
        outputs = self.fc(rnn_outputs)  # Shape: [batch_size, seq_len, vocab_size]

        return outputs

if __name__ == "__main__":
    # Hyperparameters
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    VOCAB_SIZE = 10000  # Example vocabulary size
    NUM_LAYERS = 2

    # Initialize the model
    model = CustomImageCaptioningModel(
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS
    )

    # Test the model with dummy inputs
    dummy_images = torch.randn(4, 3, 224, 224)  # Batch of 4 images
    dummy_captions = torch.randn(4, 10, EMBEDDING_DIM)  # Batch of 4 captions with max length 10

    outputs = model(dummy_images, dummy_captions)
    print("Output shape:", outputs.shape)  # Should be [batch_size, seq_len, vocab_size]
