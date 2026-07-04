# CLAUDE.md

## Project: PRAGMA Reproduction on Open Banking Data

### Mission

Reproduce the PRAGMA paper as faithfully as possible using publicly available banking and transaction datasets.

The goal is not to build a better model. The goal is to determine whether PRAGMA's reported gains arise from:

1. The architecture itself.
2. The multi-source profile + event representation.
3. Pretraining methodology.
4. Dataset-specific advantages unavailable in public data.

Success means producing rigorous evidence regarding PRAGMA's claims and understanding when those claims do or do not transfer to open datasets.

---

# Research Questions

## RQ1

Can PRAGMA be faithfully reconstructed from the paper?

Output:

* Complete architecture specification
* Reproducible implementation
* Documented assumptions where paper is ambiguous

Success Criteria:

* No undocumented implementation decisions
* Every deviation tracked in reproduction log

---

## RQ2

Does PRAGMA outperform strong tabular baselines?

Compare against:

* XGBoost
* LightGBM
* CatBoost
* Random Forest
* Logistic Regression

Success Criteria:

* Statistically significant comparison
* Multiple random seeds
* Confidence intervals reported

---

## RQ3

Does PRAGMA outperform modern deep-learning baselines?

Compare against:

* TabTransformer
* FT-Transformer
* MLP
* Transformer encoder on event sequences
* SASRec-style sequential model
* BERT4Rec-style model

Success Criteria:

* Identical train/test splits
* Same feature availability
* Comparable parameter budgets

---

## RQ4

Does profile state matter?

Ablations:

* Events only
* Profile only
* Events + profile

Success Criteria:

* Quantify incremental lift
* Identify tasks benefiting most

---

## RQ5

Does multi-source event modeling matter?

Ablations:

* Single event source
* Two event sources
* Full event set

Success Criteria:

* Marginal contribution of each source measured

---

## RQ6

Does foundation-model pretraining transfer?

Compare:

* Random initialization
* Self-supervised pretraining
* Task-specific supervised training

Success Criteria:

* Transfer lift measured across tasks

---

# Research Principles

## Principle 1

Prefer scientific validity over implementation speed.

## Principle 2

Every experiment must be reproducible.

## Principle 3

No result without a baseline.

## Principle 4

No claim without statistical evidence.

## Principle 5

Negative results are valuable.

---

# Workstreams

---

## WS1: Paper Reverse Engineering

Goal:

Recover exact PRAGMA design.

Checklist:

* [ ] Architecture diagram recreated
* [ ] Input schema documented
* [ ] Tokenization documented
* [ ] Positional encoding documented
* [ ] Pretraining objectives identified
* [ ] Loss functions identified
* [ ] Hyperparameters extracted
* [ ] Training schedule extracted
* [ ] Ambiguities documented

Deliverables:

* docs/pragma_architecture.md
* docs/reproduction_assumptions.md

Quality Gate:

A new researcher should be able to implement PRAGMA using only these documents.

---

## WS2: Dataset Construction

Goal:

Build open-data equivalent of banking customer state.

Checklist:

* [ ] Identify candidate datasets
* [ ] Create dataset inventory
* [ ] Map schema to PRAGMA schema
* [ ] Define customer profile features
* [ ] Define event streams
* [ ] Define downstream tasks
* [ ] Create train/validation/test splits
* [ ] Create reproducible data pipeline

Deliverables:

* docs/datasets.md
* datasets/processed/

Quality Gate:

Pipeline is deterministic and fully reproducible.

---

## WS3: Baseline Models

Goal:

Establish strong non-foundation baselines.

Checklist:

* [ ] Logistic Regression
* [ ] Random Forest
* [ ] XGBoost
* [ ] LightGBM
* [ ] CatBoost

Deliverables:

* benchmark tables
* tuned hyperparameters

Quality Gate:

No baseline left untuned while PRAGMA is tuned.

---

## WS4: Deep Learning Baselines

Goal:

Benchmark against modern neural methods.

Checklist:

* [ ] MLP
* [ ] TabTransformer
* [ ] FT-Transformer
* [ ] SASRec
* [ ] BERT4Rec
* [ ] Generic Transformer Encoder

Deliverables:

* benchmark tables
* reproducible configs

Quality Gate:

All models trained under comparable budgets.

---

## WS5: PRAGMA Reproduction

Goal:

Implement PRAGMA exactly.

Checklist:

* [ ] Input encoder
* [ ] Profile encoder
* [ ] Event encoder
* [ ] Fusion mechanism
* [ ] Pretraining objectives
* [ ] Finetuning pipeline
* [ ] Evaluation pipeline

Deliverables:

* pragma_model.py
* training scripts
* inference scripts

Quality Gate:

Implementation traceable to paper section-by-section.

---

## WS6: Pretraining

Goal:

Recreate self-supervised representation learning.

Checklist:

* [ ] Objective implemented
* [ ] Validation metrics tracked
* [ ] Representation quality evaluated
* [ ] Checkpoints versioned

Deliverables:

* pretrained checkpoints
* training logs

Quality Gate:

Pretraining reproducible from raw data.

---

## WS7: Downstream Tasks

Goal:

Evaluate transferability.

Potential Tasks:

* Customer segmentation
* Product recommendation
* Churn prediction
* Risk prediction
* Next-event prediction
* Transaction classification
* Anomaly detection

Checklist:

* [ ] Task definitions frozen
* [ ] Labels verified
* [ ] Metrics defined
* [ ] Baselines evaluated
* [ ] PRAGMA evaluated

Deliverables:

* task reports

Quality Gate:

All tasks use identical data splits.

---

## WS8: Ablations

Goal:

Identify where gains come from.

Checklist:

* [ ] Remove profile state
* [ ] Remove event history
* [ ] Remove pretraining
* [ ] Reduce event sources
* [ ] Vary model size
* [ ] Vary sequence length

Deliverables:

* ablation report

Quality Gate:

Every major design choice evaluated.

---

## WS9: Statistical Validation

Goal:

Verify significance.

Checklist:

* [ ] ≥ 5 seeds
* [ ] Confidence intervals
* [ ] Bootstrap testing
* [ ] Effect sizes

Deliverables:

* significance report

Quality Gate:

No result reported from a single seed.

---

## WS10: Final Analysis

Goal:

Answer whether PRAGMA's gains transfer.

Checklist:

* [ ] Reproduction accuracy assessed
* [ ] Baseline comparison completed
* [ ] Transferability assessed
* [ ] Failure modes identified
* [ ] Limitations documented

Deliverables:

* final_report.md
* reproduction_report.md

Quality Gate:

Conclusions supported by experimental evidence.

---

# Expected Repository Outputs

Required artifacts:

* Reproducible training pipeline
* Reproducible evaluation pipeline
* Benchmark tables
* Ablation tables
* Statistical significance analysis
* Trained checkpoints
* Reproduction report
* Final scientific conclusions

---

# Definition of Done

The project is complete only when:

1. PRAGMA has been reproduced as faithfully as possible.
2. Strong classical baselines have been evaluated.
3. Strong neural baselines have been evaluated.
4. Statistical significance has been established.
5. Sources of performance gains have been isolated.
6. A clear answer exists to:

"Do PRAGMA's reported advantages transfer to publicly available banking datasets?"
