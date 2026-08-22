# Google Colab Execution Guide for Payments Foundation Model & Adapters

Welcome! This guide outlines the exact process to execute the PRAGMA Payments Foundation Model & LoRA Adapter framework using Google Colab's free T4 GPUs.

---

## Prerequisites
- A Google Account for [Google Colab](https://colab.research.google.com/).
- A Kaggle Account for dataset downloading (`kaggle.json` API key).

---

## Step 1: Clone Repository into Google Drive
1. Open Google Colab and create a new notebook.
2. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Navigate to your drive and clone/navigate to the repo:
   ```bash
   cd /content/drive/MyDrive/
   git clone https://github.com/outlieralpha/Finai-research.git
   cd Finai-research
   ```

---

## Step 2: Sequential Execution Pipeline

Execute the provided Jupyter notebooks in order:

### 1. Data Pipeline (`notebooks/01_setup_and_data.ipynb`)
- **Action**: Open and run all cells.
- **What it does**: Processes PaySim (6.3M txns) & IEEE-CIS datasets, generates synthetic labels for multi-task learning (Churn, High-Value, Category, Spend Forecast), and exports `.parquet` files to `data/processed/`.

### 2. Pre-training Foundation Backbone (`notebooks/02_pretrain_foundation.ipynb`)
- **GPU Runtime**: Set Runtime -> Change runtime type -> **T4 GPU**.
- **Action**: Open and run all cells.
- **What it does**: Performs Masked Event Prediction on PaySim dataset to train the dual-encoder PRAGMA backbone.
- **Artifact**: Saves checkpoint to `models/pragma_pretrained_paysim.pth`.

### 3. Adapter Benchmarking (`notebooks/03_adapter_experiments.ipynb`)
- **GPU Runtime**: T4 GPU required.
- **Action**: Open and run all cells.
- **What it does**: Benchmarks LoRA ($r=8$), Head-Only, Full Fine-Tuning, and XGBoost across 5%, 25%, 100% data availability. Generates data efficiency plots and param comparison tables.
- **Artifact**: Saves efficiency plot to `results/data_efficiency_curves.png`.

---

## Step 3: Analyze Results
Check `results/` for benchmark tables and data efficiency curves. Use `src/config.py` as the single source of truth for paths and model hyperparameters across all notebooks.
