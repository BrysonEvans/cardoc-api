import torch
import torch.nn as nn
from models.cnn14 import Cnn14

class PannsClassifier(nn.Module):
    def __init__(self, num_labels, pretrained_path="pretrained/Cnn14_mAP=0.431.pth"):
        super().__init__()
        self.backbone = Cnn14()
        state = torch.load(pretrained_path, map_location="cpu")
        # If downloaded official weights, the key may be 'model'
        if "model" in state:
            state = state["model"]
        self.backbone.load_state_dict(state, strict=False)
        for p in self.backbone.parameters():
            p.requires_grad = False  # freeze backbone
        self.head = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )

    def forward(self, x):
        features = self.backbone(x)['embedding']
        logits = self.head(features)
        return logits
