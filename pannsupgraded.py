import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from models.cnn14 import Cnn14
import numpy as np

CSV_PATH = 'dataset/dataset.csv'
CLIP_DIR = 'clips'
SR = 32000
N_MELS = 64
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LABELS = [
    'alternator_whine',
    'pulley_belt_noise',
    'engine_idle',
    'rod_knock',
    'timing_chain_rattle',
    'silence'
]

AUG_PROB = 0.8
MIXED_SAMPLE_PROB = 0.3
CLIP_DURATION = 5  # seconds

# ----------- Audio Augmentation Utilities -----------
def random_time_shift(audio, shift_max=SR):
    shift = np.random.randint(-shift_max, shift_max)
    return torch.roll(audio, shifts=shift)

def random_gain(audio, gain_range=(0.7, 1.3)):
    gain = np.random.uniform(*gain_range)
    return audio * gain

def random_bg_noise(audio, all_audio_paths, sr=SR, noise_level=0.1):
    if len(all_audio_paths) < 2:
        return audio
    bg_path = random.choice(all_audio_paths)
    bg, _ = torchaudio.load(bg_path)
    bg = bg.mean(0)
    if bg.shape[0] < sr * CLIP_DURATION:
        bg = torch.nn.functional.pad(bg, (0, sr * CLIP_DURATION - bg.shape[0]))
    else:
        bg = bg[:sr * CLIP_DURATION]
    return (1 - noise_level) * audio + noise_level * bg

def mix_samples(audio1, label1, audio2, label2, alpha_range=(0.3, 0.7)):
    alpha = np.random.uniform(*alpha_range)
    mixed_audio = alpha * audio1 + (1 - alpha) * audio2
    mixed_label = torch.clamp(label1 + label2, 0, 1)
    if label1[3] or label2[3]:  # rod_knock
        mixed_audio[:SR*2] *= 1.3
    if label1[4] or label2[4]:  # timing_chain_rattle
        mixed_audio[SR:SR*3] *= 1.5
    return mixed_audio, mixed_label

# ----------- Dataset Loader -----------
class CarSoundDataset(Dataset):
    def __init__(self, df, clip_dir, labels, sr=SR, all_paths=None, augment=False):
        self.df = df.reset_index(drop=True)
        self.clip_dir = clip_dir
        self.labels = labels
        self.sr = sr
        self.augment = augment
        self.all_audio_paths = all_paths or [
            os.path.join(self.clip_dir, p) for p in df['path'].unique()
            if os.path.exists(os.path.join(self.clip_dir, p))
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.clip_dir, row['path'])
        if not os.path.exists(audio_path):
            print(f"[WARN] Missing file: {audio_path} (skipping)")
            return None

        audio, _ = torchaudio.load(audio_path)
        audio = audio.mean(0)
        if audio.shape[0] < self.sr * CLIP_DURATION:
            audio = torch.nn.functional.pad(audio, (0, self.sr * CLIP_DURATION - audio.shape[0]))
        else:
            audio = audio[:self.sr * CLIP_DURATION]

        labels = torch.tensor(row[self.labels].values.astype(np.float32))

        if self.augment and random.random() < MIXED_SAMPLE_PROB:
            for _ in range(3):
                mix_idx = random.randint(0, len(self.df) - 1)
                mix_row = self.df.iloc[mix_idx]
                mix_audio_path = os.path.join(self.clip_dir, mix_row['path'])
                if os.path.exists(mix_audio_path) and mix_audio_path != audio_path:
                    mix_audio, _ = torchaudio.load(mix_audio_path)
                    mix_audio = mix_audio.mean(0)
                    if mix_audio.shape[0] < self.sr * CLIP_DURATION:
                        mix_audio = torch.nn.functional.pad(mix_audio, (0, self.sr * CLIP_DURATION - mix_audio.shape[0]))
                    else:
                        mix_audio = mix_audio[:self.sr * CLIP_DURATION]
                    mix_labels = torch.tensor(mix_row[self.labels].values.astype(np.float32))
                    audio, labels = mix_samples(audio, labels, mix_audio, mix_labels)
                    break

        if self.augment and random.random() < AUG_PROB:
            if random.random() < 0.7:
                audio = random_time_shift(audio)
            if random.random() < 0.6:
                audio = random_gain(audio)
            if random.random() < 0.5:
                audio = random_bg_noise(audio, self.all_audio_paths)

        return audio, labels

def collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None, None
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)

# ----------- Model Definition -----------
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
        x = x.squeeze(1) if x.dim() == 3 else x  # ✅ FIX: Ensure input shape is [B, T]
        features = self.backbone.forward(x, None)['embedding']
        return torch.sigmoid(self.classifier(features))

# ----------- Training Loop -----------
df = pd.read_csv(CSV_PATH)
df['dominant'] = df[LABELS].idxmax(axis=1)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['dominant'], random_state=42)
all_audio_paths = [os.path.join(CLIP_DIR, p) for p in df['path'].unique() if os.path.exists(os.path.join(CLIP_DIR, p))]

train_dataset = CarSoundDataset(train_df, CLIP_DIR, LABELS, all_paths=all_audio_paths, augment=True)
val_dataset = CarSoundDataset(val_df, CLIP_DIR, LABELS, all_paths=all_audio_paths, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

def train():
    model = PannsChecklist(num_labels=len(LABELS), freeze_backbone=False).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            if xb is None:
                continue
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                if xb is None:
                    continue
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                out = model(xb)
                loss = criterion(out, yb)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/panns_cnn14_checklist_best_aug.pth')
            print("✅ Saved best model.")

if __name__ == "__main__":
    train()
