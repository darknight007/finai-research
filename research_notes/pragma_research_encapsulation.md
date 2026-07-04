# PRAGMA Replication: Research Encapsulation

## Abstract
This document encapsulates the research motivations, hypotheses, and architectural decisions involved in replicating the PRAGMA paper using open banking datasets (e.g., IEEE-CIS Fraud Detection). The core goal is to determine if PRAGMA's performance gains—typically observed on closed, proprietary banking data—transfer effectively to publicly available transaction records.

## 1. Research Motivation
Revolut's PRAGMA (Profile and Event Representations for Banking) demonstrated significant improvements in predictive tasks by utilizing a two-stream architecture that models both slow-changing user profiles and fast-changing transactional events, fused after self-supervised pre-training. However, the proprietary nature of their data prevents independent validation. 
This research aims to answer:
1. **Can the PRAGMA architecture be faithfully replicated?**
2. **Do the foundational gains hold true when applied to public, albeit limited, proxies of banking data like the IEEE-CIS Fraud Detection dataset?**

## 2. Core Methodologies

### A. The Data Translation (Public Proxy)
Because true banking datasets contain distinct `ClientID`s and rich cross-merchant histories, we had to adapt the IEEE-CIS dataset:
- **Pseudo-Client Generation**: We grouped transactions using stable identity markers (`card1`-`card6`, `addr1`, `addr2`, `P_emaildomain`) to construct a pseudo-ClientID.
- **Data Splitting**: Crucially, data must be split by *time* rather than randomly to avoid look-ahead bias in sequential event models. (First 70% Train, next 15% Val, last 15% Test).

### B. Baseline Benchmarking Strategy
We categorize baselines to isolate architectural gains:
- **Classical ML**: XGBoost, LightGBM, CatBoost. These are the industry standards for tabular data.
- **Tabular Deep Learning**: Using `pytorch-frame`, we rigorously benchmark standard neural models like `TabTransformer` and `FTTransformer`. This ensures any gains seen in PRAGMA aren't just artifacts of using deep learning, but specifically stem from the two-stream sequential approach.

### C. The PRAGMA Architecture
The model relies on three pillars:
1. **Profile Encoder**: An MLP operating over concatenated embeddings of static user features.
2. **Event Encoder**: A multi-head Transformer Encoder operating over the sequence of transaction vectors. It uses positional encoding and temporal padding masks.
3. **Fusion and Pre-training**: A mechanism to combine both streams. Before downstream finetuning, the model is trained using **Masked Event Prediction (MEP)**, forcing the network to infer missing transactions based on surrounding context and the user's profile.

## 3. Expected Outcomes & Ablation Scenarios
Through our provided ablation notebooks, we test specific hypotheses:
- **H1 (Profile vs Event)**: The Event stream alone should capture immediate behavioral shifts (e.g., rapid, consecutive anomalous transactions), while the Profile stream provides the baseline expectation of the user. We hypothesize the fusion outperforms either individually.
- **H2 (Impact of Pre-training)**: Models initialized with weights from the MEP pre-training phase will converge faster and achieve higher AUC on the downstream fraud task compared to randomly initialized counterparts.

## 4. Conclusion
By providing a reproducible PyTorch implementation, robust classical/DL baselines, and structured Google Colab workflows, this framework equips upcoming researchers with the exact tools needed to scrutinize, modify, and validate foundational banking models on open datasets.
