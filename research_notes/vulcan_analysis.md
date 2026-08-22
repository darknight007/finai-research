# Razorpay Vulcan: Architecture Analysis & Market Impact

## Executive Summary

On August 18, 2026, Razorpay launched **Vulcan**, India’s first purpose-built AI Foundation Model for payments, powered by AWS SageMaker (P5 instances) and NVIDIA H100 GPUs using the NeMo framework. Vulcan marks a shift in financial machine learning from isolated, single-task GBDT models (XGBoost/LightGBM) to a unified, multi-task foundation model trained across **~3 trillion data points and ~4 billion digital payments**.

---

## Technical Specifications & Architecture Breakdown

| Dimension | Specification |
|---|---|
| **Training Corpus** | ~3 Trillion data points across ~4 Billion payments |
| **Infrastructure** | AWS P5 Instances (NVIDIA H100 Tensor Core GPUs) + Amazon SageMaker |
| **Framework** | NVIDIA NeMo Framework & NeMo Guardrails |
| **Signals Processed** | ~3,000 transaction signals per inference call (device telemetry, IP reputational score, behavioral velocity, payment channel health) |
| **Inference Latency** | < 8 milliseconds |
| **Model Topology** | Multi-Task Transformer (Decoder-style Causal/Sequential Encoder with task-specific adapter heads) |

---

## Core Use Cases & Reported Gains

1. **Hyper-Precision Routing**
   - **Mechanic**: Predicts real-time payment gateway/bank success probabilities before sending auth requests.
   - **Impact**: +8–10% increase in overall payment success rates across 10M+ merchants.

2. **Cross-Payment Fraud & Anomaly Detection**
   - **Mechanic**: Sequential modeling of behavioral velocity across cards, net banking, and UPI networks.
   - **Impact**: 8x higher detection of international card fraud; 5x higher identification of fraudulent/disputed transactions without inflating false positives.

3. **Personalized Checkout & Merchant KYC**
   - **Mechanic**: Surfaces preferred payment apps dynamically (Magic Checkout).
   - **Impact**: +40% increase in shoppers seeing their preferred UPI app; merchant onboarding reduced from days to minutes.

---

## Strategic Implications for the Fintech Landscape

```
┌─────────────────────────────────────────────────────────┐
│                    RAZORPAY VULCAN                      │
│        (3 Trillion Signals / 4 Billion Payments)        │
├─────────────────┬───────────────────┬───────────────────┤
│  Routing Head   │    Fraud Head     │   Checkout Head   │
└────────┬────────┴─────────┬─────────┴─────────┬─────────┘
         │                  │                   │
         ▼                  ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ +8-10% Success  │ │ 8x Intl Fraud   │ │ +40% Preferred  │
│     Rate        │ │   Detection     │ │   UPI Checkout  │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

1. **Horizontal Intelligence Layer vs. Point Solutions**: Traditional payment setups maintain separate models for fraud, routing, and churn. Vulcan proves that a single shared sequence backbone yields superior inductive transfer across all tasks.
2. **The "Democratization Gap"**: While Razorpay and Nubank can deploy tens of millions of dollars into foundation model pre-training, mid-sized fintechs and regional banks cannot.
3. **The Adapter Solution**: For smaller institutions, low-rank adapters (LoRA) on pre-trained transaction backbones offer a plug-and-play alternative to achieve 95%+ of foundation model performance at < 1% of the compute cost.
