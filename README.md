# CAGA-SGA
This is the official implementation of the CAGA-SGA model, and the related paper has been submitted to IEEE TAFFC.


## 1. Overview

This codebase is designed for cross-dataset EEG emotion recognition. The training script uses source-domain EEG samples for supervised learning and target-domain EEG samples for adaptation and evaluation.

The main workflow is:

```text
prepare dataset
load one target-subject fold
train with source-domain and target-domain batches
evaluate on the target-domain test split
save the best result of each target subject
```



---

## 2. Repository Structure

```text
CAGA-SGA/
├── README.md
├── main.py              # Main training and evaluation script
├── model.py             # Main network implementation
├── layers.py            # Graph stream, attention layers, GRL, domain classifier
├── graph_align.py       # Semantic graph alignment module
└── golden_style.py      # Golden style bank and style alignment module
```

---

## 3. Environment

Create a conda environment:

```bash
conda create -n caga python=3.10 -y
conda activate caga

```

Install the required packages according to the imports in the code:

```bash
pip install numpy pandas scikit-learn
pip install torch torchvision torchaudio
pip install torch_geometric
```

If `torch_geometric` reports CUDA-related errors, install the PyG version that matches your local PyTorch and CUDA versions.

The main imported packages are:

```text
numpy
pandas
scikit-learn
torch
torch_geometric
```

---

## 4. Dataset

This repository is intended for EEG emotion recognition experiments on the SEED series datasets.

Official SEED dataset website:

```text
https://bcmi.sjtu.edu.cn/home/seed/
```

The related datasets include:

| Dataset | Emotion categories |
|---|---|
| SEED | positive, negative, neutral |
| SEED-IV | happy, sad, fear, neutral |
| SEED-V | happy, sad, fear, disgust, neutral |
| SEED-VII | happy, sad, fear, disgust, neutral, anger, surprise |


---

## 5. Run

After preparing the dataset, run:

```bash
python main.py
```

The script will iterate over target subjects, train the model, evaluate on the target-domain data, and save the result files under:

```text
./result/
```

---

## 6. Hyperparameters

The reproduction hyperparameters are listed below.

| Hyperparameter | Value |
|---|---:|
| Batch Size | 48 |
| Learning Rate | $5\times10^{-4}$ |
| Weight Decay | $1\times10^{-4}$ |
| Max Iters | 1000 |
| Optimizer | AdamW |
| Dropout Rate | 0.3 |
| Feature Dimension $d_{\mathrm{model}}$ | 64 |
| CAGA Layers $L$ | 3 |
| Attention Heads | 4 |
| Top-k Sparsification | 8 |
| GCN Hidden Dimension | 64 |
| $\lambda_{\mathrm{dis}}$ | 0.1 |
| $\lambda_{\mathrm{style}}$ | 0.1 |
| $\lambda_{\mathrm{pres}}$ | 1.0 |
| $\lambda_{\mathrm{gold}}$ | 1.0 |
| $\lambda_{\mathrm{align}}$ | 1.0 |
| $\lambda_{\mathrm{gram}}$ | 0.1 |
| $\lambda_e$ | 1.0 |
| $\lambda_v$ | 1.0 |
| Pseudo-label Threshold $\delta$ | 0.90 |
| GRL Loss Cap $\tau$ | 1.0 |
| Numerical Stability $\epsilon$ | $1\times10^{-5}$ |
