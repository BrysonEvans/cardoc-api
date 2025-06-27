#import torch.nn as nn
from models.cnn14 import Cnn14

SR = 32000
N_MELS = 64

class PannsChecklist(nn.Module):
    def __init__(self, num_labels, freeze_backbone=False):
        super().__init__()
        self.backbone = Cnn14(
            sample_rate=SR, window_size=1024, hop_size=320,
            mel_bins=N_MELS, fmin=50, fmax=14000, classes_num=527
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )

    def forward(self, x):
        x = x.squeeze(1) if x.dim() == 3 else x
        features = self.backbone.forward(x, None)['embedding']
        return torch.sigmoid(self.classifier(features))

print("✅ Using correct PannsChecklist with forward()")
# force redeploy
