import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torchvision.models import resnet50

class MultiModalModel(nn.Module):
    def __init__(self):
        super(MultiModalModel, self).__init__()
        # Image Encoder (ResNet)
        self.image_encoder = resnet50(pretrained=True)
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, 512)

        # Text Encoder (BERT)
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.text_fc = nn.Linear(768, 512)

        # Fusion Layer
        self.fusion = nn.Linear(512 * 2, 256)
        self.classifier = nn.Linear(256, 10)  # Example: 10 classes

    def forward(self, image, text_input_ids):
        img_features = self.image_encoder(image)
        text_outputs = self.text_encoder(input_ids=text_input_ids)
        text_features = self.text_fc(text_outputs.pooler_output)
        combined = torch.cat((img_features, text_features), dim=1)
        fusion_output = torch.relu(self.fusion(combined))
        return self.classifier(fusion_output)
