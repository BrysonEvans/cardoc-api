"""
panns_infer.py  –  inference-only version for Render
• No pandas / CSV / scikit-learn
• Same PannsChecklist API you already use
"""

from pathlib import Path
import torch
import torch.nn as nn
from models.cnn14 import Cnn14   # same backbone you trained with

# ----------------------------------------------------------------------
# Constants (copied from training file)
# ----------------------------------------------------------------------
SR          = 32_000
N_MELS      = 64
LABELS      = [
    "alternator_whine",
    "pulley_belt_noise",
    "engine_idle",
    "rod_knock",
    "timing_chain_rattle",
    "silence",
]
NUM_LABELS  = len(LABELS)


# ----------------------------------------------------------------------
# Model definition  (unchanged except for docstring tweaks)
# ----------------------------------------------------------------------
class PannsChecklist(nn.Module):
    """
    Forward-only checklist model.
    """

    def __init__(self, num_labels: int = NUM_LABELS, freeze_backbone: bool = True):
        super().__init__()

        self.backbone = Cnn14(
            sample_rate=SR,
            window_size=1024,
            hop_size=320,
            mel_bins=N_MELS,
            fmin=50,
            fmax=14_000,
            classes_num=527,
        )
        if freeze_backbone:
            self.backbone.requires_grad_(False)

        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels),
        )

    # --------------------------------------------------------------
    # Helper for Render: load weights only
    # --------------------------------------------------------------
    @classmethod
    def load_from_file(cls, ckpt_path: str | Path):
        model = cls()
        ckpt  = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        model.load_state_dict(ckpt, strict=False)
        model.eval()
        return model

    # --------------------------------------------------------------
    # Inference helper used by app.py
    # --------------------------------------------------------------
    @torch.no_grad()
    def predict(self, waveform, sr: int):
        """
        Args
        ----
        waveform : 1-D float32 numpy or torch array
        sr       : sample-rate of waveform
        Returns
        -------
        dict {label: probability}
        """
        import torchaudio
        import numpy as np

        if isinstance(waveform, np.ndarray):
            waveform = torch.tensor(waveform)

        if sr != SR:
            waveform = torchaudio.functional.resample(waveform, sr, SR)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (C, T)

        feats = self.backbone.forward(waveform, None)["embedding"]
        probs = torch.sigmoid(self.classifier(feats)).squeeze().tolist()

        return dict(zip(LABELS, probs))
