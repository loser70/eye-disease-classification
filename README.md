# 👁️ Eye Disease Classification — 3-Model Ensemble

A deep learning pipeline for classifying **10 retinal diseases** from fundus images, built with PyTorch and designed to run on Kaggle.

The ensemble combines three complementary backbones — each capturing different visual patterns — and averages their softmax outputs for a more robust final prediction.

| Backbone | Inductive Bias | Input Size |
|---|---|---|
| EfficientNetV2-L + CBAM | Efficient CNN scaling + attention | 480 × 480 |
| ConvNeXt-Large | Modern CNN with depthwise convolutions | 384 × 384 |
| ViT-Large/16 | Global self-attention via Vision Transformer | 224 × 224 |

---

## Classes

| Label | Disease |
|---|---|
| 0 | Central Serous Chorioretinopathy |
| 1 | Diabetic Retinopathy |
| 2 | Disc Edema |
| 3 | Glaucoma |
| 4 | Healthy |
| 5 | Macular Scar |
| 6 | Myopia |
| 7 | Pterygium |
| 8 | Retinal Detachment |
| 9 | Retinitis Pigmentosa |

---

## Architecture Highlights

**CBAM Attention** — plugged into EfficientNetV2-L to weight both channel-wise and spatial features, helping the model focus on diagnostically relevant retinal regions.

**Mixed-size inputs** — each backbone uses its own optimal resolution, so no single model is forced into a suboptimal crop.

**Gradient accumulation + AMP** — trains effectively with small batch sizes (8–16) on a single Kaggle GPU while maintaining stable gradient estimates.

**Cosine annealing with warm restarts** — scheduler with early stopping (patience = 7) to avoid overfitting on small medical datasets.

---

## Requirements

```
torch
torchvision
timm
numpy
pandas
matplotlib
seaborn
scikit-learn
tqdm
Pillow
```

Install extras (already available on Kaggle):
```bash
pip install timm -q
```

---

## Dataset Structure

The notebook auto-detects `train/`, `val/`, and `test/` splits anywhere under `/kaggle/input`. Expected layout:

```
dataset/
├── train/
│   ├── Glaucoma/
│   ├── Diabetic Retinopathy/
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

Each class folder should contain `.jpg`, `.jpeg`, or `.png` images.

---

## Training

The notebook trains each model independently with:

- **Optimizer:** AdamW (`lr=1.2e-4`, `weight_decay=3e-5`)
- **Loss:** CrossEntropyLoss with label smoothing (0.1)
- **Epochs:** 15 with early stopping
- **Gradient clipping:** 0.8
- **Mixed precision:** enabled automatically if GPU is available

Best model weights are saved per backbone. Final predictions are the average of all three models' softmax outputs.

---

## Notebook Structure

| Cell | Description |
|---|---|
| 1 | Imports & device setup |
| 2 | Dataset path resolution |
| 3 | Class mapping & DataFrames |
| 4 | Dataset class & DataLoaders |
| 5 | CBAM attention module |
| 6 | Model definitions (EfficientNet, ConvNeXt, ViT) |
| 7 | Hyperparameters |
| 8 | Generic training function |
| ... | Training loop, evaluation, ensemble inference |

---

## Usage on Kaggle

1. Upload the notebook to a Kaggle kernel
2. Add your fundus image dataset as an input
3. Enable GPU (P100 or T4 recommended)
4. Run all cells

Outputs (model weights, metrics) are saved to `/kaggle/working`.

---

## License

MIT
