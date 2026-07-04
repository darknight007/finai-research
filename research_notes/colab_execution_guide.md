# Google Colab Execution Guide for PRAGMA Replication

Welcome! This guide outlines the exact, step-by-step process for an upcoming researcher to clone, setup, and execute the PRAGMA replication framework using Google Colab. Colab provides free T4 GPUs which are perfect for testing these deep learning and tabular models.

## Prerequisites
- A Google Account to access [Google Colab](https://colab.research.google.com/).
- A Kaggle Account to download the IEEE-CIS Fraud Detection dataset (ensure you have generated a `kaggle.json` API token).
- Basic familiarity with Jupyter Notebooks and running Python cells.

## Step 1: Clone the Repository into Colab
1. Open Google Colab and click **File > New notebook**.
2. Mount your Google Drive so your progress is saved persistently.
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Navigate to a directory in your drive and clone the project repository (assuming this code is hosted on GitHub, replace with actual URL, or upload the zipped codebase):
   ```bash
   cd /content/drive/MyDrive/
   git clone <YOUR_REPO_URL> Finai-research
   cd Finai-research
   ```

## Step 2: Set Up Kaggle for Data Ingestion
Colab needs your Kaggle API key to download the dataset automatically.
1. Upload your `kaggle.json` to the Colab environment.
2. Run the following cell to set it up:
   ```bash
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

## Step 3: Run the Pipeline (Sequentially)
Colab has an execution limit, but each step is designed to save its artifacts (datasets, models) to disk. Proceed by opening each of the provided notebooks sequentially:

### A. Data Preparation (`notebooks/data_pipeline.ipynb`)
- **Action**: Open the notebook and run all cells.
- **What it does**: Downloads the raw IEEE-CIS dataset, creates pseudo-ClientIDs, processes features, and outputs train/val/test `.parquet` files into the `data/processed/` folder.

### B. Evaluate Classical Baselines (`notebooks/classical_baselines.ipynb`)
- **Action**: Open the notebook and run all cells.
- **What it does**: Installs XGBoost, LightGBM, CatBoost and trains the tree models. It will print the validation and test AUC metrics.

### C. Evaluate Deep Learning Baselines (`notebooks/dl_baselines.ipynb`)
- **Important**: Ensure your runtime is set to GPU. Go to **Runtime > Change runtime type** -> select **T4 GPU**.
- **Action**: Open the notebook and run all cells.
- **What it does**: Installs `pytorch-frame`, materializes the tensor datasets, and trains TabTransformer, FTTransformer, and ResNet representations. 

### D. Self-Supervised Pre-Training (`notebooks/pretraining.ipynb`)
- **Important**: Ensure your GPU runtime is active.
- **Action**: Open the notebook and run all cells.
- **What it does**: Initializes the PRAGMA dual-encoder architecture and trains the Masked Event Prediction task.
- **Output**: Saves the foundational representation weights to `models/pragma_pretrained.pth`.

### E. Fine-Tuning and Ablations (`notebooks/finetuning_and_evaluation.ipynb` & `notebooks/ablations.ipynb`)
- **Action**: Open the notebooks and run all cells.
- **What it does**: Loads the `.pth` weights, replaces the pre-training head with a classification head, and finetunes on the fraud detection task. Evaluates performance contributions by disabling profile or event streams (ablations).

## Step 4: Analyze Results
After completing the notebooks, collate the AUC and F1 scores printed at the end of each stage. Compare the Classical Baselines vs DL Baselines vs PRAGMA. Use these results to answer the research questions in the summary report.
