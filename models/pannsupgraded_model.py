import torch
import torch.nn as nn
from models.cnn14 import Cnn14

SR = 32000
N_MELS = 64
CLIP_DURATION = 5  # seconds

LABELS = [
    'alternator_whine',
    'pulley_belt_noise',
    'engine_idle',
    'rod_knock',
    'timing_chain_rattle',
    'silence'
]

class PannsChecklist(nn.Module):
    def __init__(self, num_labels=len(LABELS), freeze_backbone=False):
        super().__init__()
        self.backbone = Cnn14(
            sample_rate=SR,
            window_size=1024,
            hop_size=320,
            mel_bins=N_MELS,
            fmin=50,
            fmax=14000,
            classes_num=527
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
            nn.Linear(512, num_labels),
            nn.Sigmoid()  # final output: multi-label [0..1]
        )

    def forward(self, waveform):
        """
        waveform: Tensor [batch, time]
        returns: Tensor [batch, num_labels]
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # [B, 1, T]

        # Backbone expects [B, T], no channels dimension
        B = waveform.shape[0]
        x = waveform.squeeze(1)  # [B, T]
        
        # Get embeddings from backbone
        features = self.backbone.forward(x, None)['embedding']  # [B, 2048]

        # Classifier head
        out = self.classifier(features)  # [B, num_labels]
        return out
