# AI-Powered Waste Segregation System

Real-time waste classification system using YOLOv8 and OpenCV that categorizes waste into three classes for automated segregation.

## Categories

| Class | Items | Display Color |
|-------|-------|---------------|
| Paper / Cardboard | Paper, cardboard | Orange |
| Plastic | Plastic bottles, bags, containers | Green |
| Miscellaneous | Glass, metal, general trash | Red |

## Dataset

- **Source**: [TrashNet](https://github.com/garythung/trashnet) (6 classes remapped to 3)
- **Total images**: 2,527
- **Split**: 70% train (1,767) / 15% val (378) / 15% test (382)
- **Class mapping**:
  - `paper` + `cardboard` → `paper_cardboard`
  - `plastic` → `plastic`
  - `glass` + `metal` + `trash` → `miscellaneous`

## Training Results

**Model**: YOLOv8n-cls (fine-tuned) | **Epochs**: 50 | **Optimizer**: Adam (lr=0.001) | **Image size**: 224x224

| Metric | Value |
|--------|-------|
| **Val Accuracy** | 95.8% |
| **Test Accuracy** | 94.5% |
| **Final Train Loss** | 0.031 |
| **Final Val Loss** | 0.599 |

### Training Progress

| Epoch | Train Loss | Val Acc | Val Loss |
|-------|-----------|---------|----------|
| 1     | 0.669     | 79.6%   | 0.783    |
| 5     | 0.281     | 88.4%   | 0.673    |
| 10    | 0.184     | 89.7%   | 0.659    |
| 20    | 0.138     | 94.2%   | 0.615    |
| 30    | 0.095     | 92.3%   | 0.632    |
| 40    | 0.063     | 94.2%   | 0.605    |
| 50    | 0.031     | 95.8%   | 0.599    |

### Per-Class Accuracy (Test Set Confusion Matrix)

| Class | Correct | Total | Accuracy |
|-------|---------|-------|----------|
| Miscellaneous | 153 | 157 | 97.5% |
| Paper/Cardboard | 145 | 149 | 97.3% |
| Plastic | 64 | 72 | 88.9% |

## Setup

```bash
# Create conda environment
conda create -n waste_segregation python=3.10 -y
conda activate waste_segregation

# Install dependencies
pip install -r requirements.txt

# Download and prepare dataset
python prepare_dataset.py

# Train model (50 epochs)
python train.py

# Run inference
python inference.py --mode webcam          # Real-time webcam
python inference.py --mode image --source <path>  # Single image
```

## Project Structure

```
Waste-Segregation/
├── config.yaml          # All hyperparameters and paths
├── prepare_dataset.py   # Downloads TrashNet, remaps classes, splits data
├── train.py             # Fine-tunes YOLOv8n-cls
├── inference.py         # Webcam real-time + static image inference
├── utils.py             # Shared constants and config loader
├── requirements.txt     # Python dependencies
└── model/
    └── best.pt          # Trained model weights
```

## Tech Stack

- **Model**: YOLOv8n-cls (Ultralytics)
- **Framework**: PyTorch 2.1.2
- **Computer Vision**: OpenCV 4.9
- **Python**: 3.10
