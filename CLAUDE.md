# Grasp Detection Research Paper Project

## Project Goal
Produce a research paper applying deep learning to robotic grasp detection. This is a course research paper for a Columbia Business School class. The deliverables are:
1. Working code that trains a CNN to predict grasp configurations from RGB images
2. A research paper (~8-12 pages) reporting methodology and results

**Time budget: ~1 day total. Prioritize getting an end-to-end pipeline working over fancy improvements.** If something is taking too long, simplify.

## Research Question
Can a pretrained CNN (ResNet-50) effectively predict robotic grasp rectangles from RGB images, and how does ImageNet-pretrained initialization compare to training from scratch on the same task?

## Anchor Paper
Redmon, J., & Angelova, A. (2015). "Real-Time Grasp Detection Using Convolutional Neural Networks." ICRA 2015.
- arXiv: https://arxiv.org/abs/1412.3128

We replicate the single-stage CNN regression formulation, with these adaptations:
- Use ResNet-50 instead of AlexNet
- Compare pretrained vs from-scratch initialization
- Use Cornell Grasping Dataset standard evaluation metric

## Dataset
**Cornell Grasping Dataset**
- Kaggle mirror (preferred): https://www.kaggle.com/datasets/oneoneliu/cornell-grasp
- Original: http://pr.cs.cornell.edu/grasping/rect_data/data.php
- ~885 RGB-D images of 240 graspable objects
- Each image annotated with multiple positive (`pcd*cpos.txt`) and negative (`pcd*cneg.txt`) grasp rectangles
- Each rectangle = 4 corner points (x, y) representing an oriented rectangle

### Grasp Representation
Convert 4-corner annotations to 5-parameter form:
- (x, y): center of rectangle
- θ: orientation in radians
- w: width (gripper opening)
- h: height (gripper plate width)

## Task Formulation
- Input: 224×224 RGB image (resize from original 640×480)
- Output: 5-dim grasp parameters (x, y, θ, w, h)
- Loss: L2 regression (normalize each dimension before computing loss so they contribute on similar scale)

For training images with multiple positive grasps, randomly sample one per epoch. For evaluation, prediction is "correct" if it matches ANY ground-truth grasp by the Cornell metric.

## Model Architecture
```python
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class GraspNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)
        self.backbone.fc = nn.Linear(2048, 5)  # x, y, θ, w, h

    def forward(self, x):
        return self.backbone(x)
```

## Training Spec
- Optimizer: Adam, lr=1e-4, weight_decay=1e-5
- Batch size: 32 (reduce if OOM)
- Epochs: 30 max with early stopping (patience=5 on val loss)
- Train/val/test split: 70/15/15 (random by image — fine for this scope)
- Augmentation: random crop, horizontal flip, small rotation (±15°), color jitter
- Normalize images with ImageNet mean/std

## Evaluation Metric
**Cornell standard metric** — a predicted grasp is correct iff BOTH:
1. Jaccard index (IoU) with any ground-truth positive grasp > 0.25
2. Angle difference with that same grasp < 30°

Report:
- **Grasp accuracy** = % of test images with a correct prediction (primary)
- **MAE on individual grasp parameters** (secondary, for completeness)

Use `shapely.geometry.Polygon` for oriented-rectangle IoU computation.

## Two Experiments to Run
1. **Pretrained**: ResNet-50 + ImageNet weights, fine-tuned on Cornell
2. **From scratch**: ResNet-50 with random init, trained on Cornell

Same hyperparameters for both. Report side-by-side comparison.

## Repository Structure
```
grasp-detection-paper/
├── README.md
├── requirements.txt
├── .gitignore
├── CLAUDE.md (this file)
├── data/                    # Cornell dataset (gitignored)
├── src/
│   ├── __init__.py
│   ├── dataset.py           # Cornell dataset class + parsing
│   ├── model.py             # GraspNet definition
│   ├── train.py             # training loop with CLI args
│   ├── evaluate.py          # Cornell metric evaluation
│   └── utils.py             # grasp rectangle utilities (IoU, angle conversion)
├── scripts/
│   ├── download_data.sh
│   ├── run_pretrained.sh
│   └── run_scratch.sh
├── results/                 # checkpoints + logs (gitignored)
│   ├── pretrained/
│   └── scratch/
└── paper/
    └── paper.md             # the actual research paper
```

## Paper Structure (paper/paper.md, ~8-12 pages)
1. **Introduction** (~1 page) — problem motivation, industry context (Symbotic, Covariant, Berkshire Grey, humanoid manipulation), research question, contributions
2. **Related Work** (~1 page) — Lenz/Lee/Saxena (2015), Redmon & Angelova (2015), brief mention of GG-CNN / GR-ConvNet
3. **Method** (~1.5 pages) — 5-parameter grasp formulation, ResNet-50 + regression head, loss + training
4. **Experiments** (~2-3 pages) — Cornell dataset, splits, implementation details, Cornell metric, results table (pretrained vs scratch), qualitative grasp visualizations on test images
5. **Discussion** (~1-2 pages) — what the pretrained-vs-scratch gap reveals about transfer learning for robotics, failure mode analysis, connection to industry challenges (sim-to-real, generalization)
6. **Conclusion** (~0.5 page) — summary, limitations, future directions

## Workflow Order (follow this sequence)
1. Set up repo structure and verify dependencies install
2. Download dataset; write parser for the `pcd*cpos.txt` annotation format
3. Implement `Dataset` class with augmentations; verify with a quick visualization
4. Implement model; verify forward pass on a dummy batch
5. Implement training loop with logging (loss curves saved to disk)
6. Run pretrained variant for ~3 epochs as a smoke test (loss should decrease)
7. Implement Cornell-metric evaluation
8. Run full training for both pretrained and scratch
9. Generate result plots, qualitative visualizations
10. Write paper sections referencing actual numbers from results

## Execution Environment: Colab

**Code is written locally; training runs on Google Colab (free tier T4 GPU).**

Implications for how code should be structured:
- All training and evaluation must runnable as `python -m src.train ...` and `python -m src.evaluate ...` (CLI entry points), so the Colab notebook can call them with `!` shell commands
- `data/` and `results/` will be symlinked to Google Drive in Colab so they persist across session disconnects — code should not assume these are local
- `scripts/download_data.sh` should be **idempotent**: skip download if data already exists. The script will be re-run on every Colab session
- Checkpoints saved to `results/<variant>/best.pth` automatically end up on Drive via the symlink
- Logs (loss curves) should be written to disk (e.g., `results/<variant>/train.log` or a JSON), not just stdout, since Colab cells can lose output on disconnect
- The orchestrating notebook is `colab_train.ipynb` at the repo root — it handles Drive mount, repo clone, symlinking, dataset download, and training calls

**Workflow:**
1. Develop code locally with Claude Code
2. `git push` to GitHub
3. Open `colab_train.ipynb` in Colab; it pulls latest code and runs training
4. Checkpoints + logs end up on Drive; pull them locally to write paper

## Constraints / Things to Watch
- **Get end-to-end pipeline working first.** Even if metrics are bad, having a working pipeline is more important than tuning.
- **Start with small data.** Run training on 50 images first to verify pipeline before training on full dataset.
- **Save checkpoints.** Best model on val loss; save to `results/<variant>/best.pth`.
- **Save training logs to disk** (Colab disconnects can wipe stdout).
- **Cornell annotation parser.** Read the dataset README carefully — the `.txt` file format matters. Each line in `pcd*cpos.txt` is one (x, y) corner; rectangles are groups of 4 consecutive lines.
- **`download_data.sh` must be idempotent** — check if data exists before downloading.
- **Don't over-engineer.** This is a 1-day project. Skip multi-grasp prediction, depth fusion, advanced augmentations.
