# Grasp Detection with Pretrained CNNs

Research paper project. Replicates Redmon & Angelova (2015), "Real-Time Grasp Detection Using Convolutional Neural Networks," using ResNet-50 on the Cornell Grasping Dataset, comparing ImageNet-pretrained vs from-scratch initialization.

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset

Download the Cornell Grasping Dataset and extract to `data/cornell/`. See `scripts/download_data.sh`.

## Run

```bash
# Train with pretrained ResNet-50
python -m src.train --pretrained --output results/pretrained

# Train from scratch
python -m src.train --output results/scratch

# Evaluate on test set
python -m src.evaluate --checkpoint results/pretrained/best.pth
```

## Paper

See `paper/paper.md`.
