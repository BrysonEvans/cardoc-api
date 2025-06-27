# stage1_model.py

import torch
import torchaudio
from models.cnn14 import Cnn14

SR = 32000
N_MELS = 64

class Stage1EngineDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Cnn14(
            sample_rate=SR, window_size=1024, hop_size=320,
            mel_bins=N_MELS, fmin=50, fmax=14000, classes_num=527
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 1)  # Single output: engine vs non_engine
        )

    def forward(self, x):
        x = x.squeeze(1) if x.dim() == 3 else x
        features = self.backbone.forward(x, None)['embedding']
        return self.classifier(features)
