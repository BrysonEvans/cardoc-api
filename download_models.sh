#!/bin/bash
set -e

# Make sure the models directory exists
mkdir -p models

# Download your main PANNs model (Stage 2)
if [ ! -f models/panns_cnn14_checklist_best_aug.pth ]; then
    echo "Downloading PANNs model..."
    gdown --id 1ZXaZA1Q9tCCjEGJMFy0Pt5QNDFsOVLuV -O models/panns_cnn14_checklist_best_aug.pth
fi

# Download stage1 model
if [ ! -f models/stage1_engine_detector.pth ]; then
    echo "Downloading Stage 1 model..."
    gdown --id 1D5r5ldFuWurM8awnquLFU7dNwu1WsTQz -O models/stage1_engine_detector.pth
fi

echo "Model downloads complete!"