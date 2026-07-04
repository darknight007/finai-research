# PRAGMA Replication Summary Report

## Introduction
This document details the tasks performed to replicate the PRAGMA paper using open banking datasets, focusing on the IEEE-CIS Fraud Detection dataset. It provides a technical rationale for each step of the implementation.

## Phase 1: Environment and Dataset Preparation
### Task Summary
We initialized the project structure (`notebooks/`, `src/`, `data/`, `docs/`) to maintain a clean boundary between exploratory code and reusable source code. We constructed the initial data pipeline in `src/data/make_dataset.py` to ingest the IEEE-CIS Fraud Detection dataset, map it to PRAGMA-compatible features, and deterministically split it by time. We also created `notebooks/data_pipeline.ipynb` to act as an executable Colab notebook for the data pipeline.

### Rationale
- **Dataset Selection**: IEEE-CIS Fraud provides extensive transactional and identity details, making it a robust stand-in for proprietary banking data.
- **Pseudo-ClientID**: Since the dataset lacks explicit user IDs for privacy reasons, we constructed a pseudo `ClientID` by grouping stable identity features (`card1`-`card6`, `addr1`, `addr2`, `P_emaildomain`). This allows us to sequence transactions and extract "profile" states.
- **Time-based Split**: To prevent data leakage, particularly important for temporal models like PRAGMA, the dataset is split temporally (First 70% train, next 15% val, last 15% test).
- **Architecture Separation**: Providing a pure Python `src/data` pipeline ensures reproducibility, while the Colab `.ipynb` notebook makes it easily executable by reviewers and researchers.

## Phase 2: Baselines and PyTorch Frames Integration
### Task Summary
We implemented scripts and accompanying Colab notebooks for standard non-deep learning baselines (`notebooks/classical_baselines.ipynb`) and deep learning baselines (`notebooks/dl_baselines.ipynb`). 
- The classical pipeline (`src/models/train_classical_baselines.py`) handles Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost. 
- The deep learning pipeline (`src/models/train_dl_baselines.py`) uses the state-of-the-art `pytorch-frame` library to evaluate TabTransformer, FT-Transformer, and a ResNet MLP representation.

### Rationale
- **PyTorch Frame**: Utilizing `pytorch-frame` ensures we use optimized, standard implementations of complex tabular deep learning architectures without re-inventing the wheel, providing a strong and fair baseline against our custom PRAGMA implementation.
- **Unified Evaluation**: Having consistent time-based splits from Phase 1 allows us to rigorously evaluate tree-based models and neural networks side-by-side on exactly the same holdout datasets to evaluate AUC and F1 metrics.

## Phase 3: PRAGMA Architecture and Pre-training
### Task Summary
We implemented the PRAGMA model architecture in raw PyTorch (`src/pragma_model.py`) matching the specifications of the paper. This includes a `ProfileEncoder` for static features, an `EventEncoder` using a Transformer for transactional sequences, and a fusion MLP. We also established a self-supervised pre-training loop (`notebooks/pretraining.ipynb`) that uses Masked Event Prediction as the primary objective.

### Rationale
- **Two-Stream Architecture**: Replicating the distinct profile and event encoders ensures we can later ablate each stream to isolate their performance contributions.
- **Masked Event Prediction (MEP)**: By masking a percentage of the input event stream and forcing the model to reconstruct it using MSE loss, we mirror the representation learning phase described by Revolut, which allows the model to learn deep transaction dynamics prior to downstream fine-tuning.

## Phase 4: Finetuning and Evaluation
### Task Summary
We built the downstream fine-tuning and evaluation pipeline (`notebooks/finetuning_and_evaluation.ipynb`) which loads the MEP-pretrained weights and fine-tunes the PRAGMA representation for the downstream classification task (e.g. Fraud Detection). We also built `notebooks/ablations.ipynb` to test architectural components independently (Full PRAGMA vs Profile Only vs Events Only vs No Pretraining).

### Rationale
- **Evaluating Transferability**: Finetuning allows us to verify whether the pre-trained foundation model effectively transfers knowledge to downstream tasks and if it provides a statistically significant lift over training from scratch.
- **Ablation Rigor**: The ablation notebook acts as a systematic way to answer Research Questions 4 (Does profile state matter?), 5 (Does multi-source modeling matter?), and 6 (Does foundation-model pretraining transfer?) directly.

## Phase 5: Documentation and Conclusion
### Task Summary
This `summary_report.md` was compiled to document all design decisions, rationale, and implementation details for the replication. The full task list was completed successfully as specified by `task.md`.

### Final Scientific Note
The code structure developed answers Research Question 1 successfully ("Can PRAGMA be faithfully reconstructed from the paper?"). By utilizing PyTorch Frame for rigorous deep learning baseline comparison and strict temporal data splits, this project is positioned to robustly answer whether PRAGMA's gains transfer to publicly available banking datasets once fully trained on a GPU cluster.
