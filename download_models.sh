#!/bin/bash
set -e

# Make sure the models directory exists
mkdir -p models

# Download your main PANNs model (replace the ID if needed)
if [ ! -f models/panns_cnn14_checklist_best_aug.pth ]; then
    echo "Downloading PANNs model..."
    gdown --id 1ZXaZA1Q9tCCjEGJMFy0Pt5QNDFsOVLuV -O models/panns_cnn14_checklist_best_aug.pth
fi

# Download stage1 model if needed (uncomment and fill in the correct ID)
# if [ ! -f models/stage1_engine_detector.pth ]; then
#     echo "Downloading Stage 1 model..."
#     gdown --id <YOUR_STAGE1_ID> -O models/stage1_engine_detector.pth
# fi

echo "Model downloads complete!"
