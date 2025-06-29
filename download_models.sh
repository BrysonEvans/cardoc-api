import os
import gdown

os.makedirs("models", exist_ok=True)
# Replace with your actual file ID if different
file_id = "1ZXaZA1Q9tCCjEGJMFy0Pt5QNDFsOVLuV"
output = "models/panns_cnn14_checklist_best_aug.pth"

if not os.path.exists(output):
    print("Downloading model weights from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
else:
    print("Model already exists, skipping download.")
