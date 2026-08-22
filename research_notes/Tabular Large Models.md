PRAGMA's claim is essentially that existing work solved **parts** of the banking-foundation-model problem, but nobody had combined:

1. **Static customer profile data** (demographics, account metadata, risk attributes),  
2. **Multiple event streams** (transactions, card activity, app usage, transfers, etc.),  
3. **A single transferable backbone** usable across many downstream banking tasks.

Here's a concise breakdown of the cited papers and how they differ.

| Paper | Main Idea | Data Representation | Key Finding | Why PRAGMA says it's insufficient |
| ----- | ----- | ----- | ----- | ----- |
| TabTransformer (2020) | Transformer over categorical table columns | Single tabular row | Contextual embeddings of categorical features outperform classical tabular methods in many settings | No temporal behavior, no sequences, no multi-source events |
| FT-Transformer (2021) | Treat every table feature as a token | Single tabular row | Strong general-purpose transformer baseline for tabular ML | Still fixed-schema rows; no transaction histories |
| SASRec (2018) | Self-attention for user interaction sequences | Ordered item history | Learns user preferences from recent sequence context | Only sequential events; no static profile state |
| BERT4Rec (2019) | BERT-style masked sequence modeling | Interaction histories | Bidirectional context improves recommendation quality | Focused on recommendation sequences, not heterogeneous banking data |
| FinBERT (2020) | Financial-domain language model | Financial text | Domain-specific pretraining improves finance NLP tasks | Text-centric, not transaction-centric |
| Time-Series Foundation Models (2023–2024 family) | Pretrain on generic temporal signals | Time-series tokens | Transfer learning works for forecasting and temporal tasks | Usually numerical time series; limited customer-state modeling |
| nuFormer (2025) | Transformer over transaction ledger | Transaction events | Learns useful user representations from banking activity for recommendations | Uses mainly one event source and recommendation-focused evaluation |
| TransactionGPT (2025) | Generative model over transaction histories | Transaction sequences | Can model trajectories, generate future behavior, detect anomalies | Focuses on ledger events; lacks rich profile-state integration and broad task transfer |

---

## **1\. TabTransformer (Huang et al., 2020\)**

### **Approach**

* Each categorical column becomes a token.  
* Transformer learns interactions among columns.  
* Numerical features are appended later.

### **Key finding**

Column relationships matter. Contextualized embeddings improve robustness and predictive performance on tabular classification tasks.

### **Limitation relative to PRAGMA**

A customer is represented as **one row**, not a history of behavior. There is no notion of time or transactions.

---

## **2\. FT-Transformer (Gorishniy et al., 2021\)**

### **Approach**

* Every feature (categorical and numerical) becomes a token.  
* Pure Transformer architecture for tabular data.

### **Key finding**

One of the strongest deep-learning baselines for tabular prediction and often competitive with gradient boosting.

### **Limitation relative to PRAGMA**

Still assumes a fixed-schema record. It models feature interactions, not behavioral sequences.

---

## **3\. SASRec (Kang et al., 2018\)**

### **Approach**

* Applies self-attention to a user's sequence of interactions.  
* Similar to Transformer encoder but focused on recommendation.

### **Key finding**

Attention identifies which past interactions are most relevant for predicting the next one.

### **Limitation relative to PRAGMA**

Only sequence data exists. User profile attributes are largely absent.

Think:

\[Amazon item A\] \-\> \[item B\] \-\> \[item C\]

rather than

Profile \+ Transactions \+ App Usage \+ Cards \+ Transfers

---

## **4\. BERT4Rec (Sun et al., 2019\)**

### **Approach**

* Uses BERT-style masked-item prediction.  
* Learns bidirectional representations of interaction histories.

### **Key finding**

Looking both backward and forward in a sequence improves recommendation quality over autoregressive approaches.

### **Limitation relative to PRAGMA**

Still a recommendation model over item histories, not a universal customer representation model.

---

## **5\. Financial Foundation Models (2020–2024)**

The papers cited by PRAGMA (Yang et al., Wu et al., Yang et al., Jin et al., Ansari et al.) generally fall into two groups:

### **Financial NLP models**

Examples:

* Financial BERT variants  
* Financial LLMs

**Finding:** Domain-specific pretraining improves financial text understanding.

**Limitation:** Transaction ledgers are not the primary modality.

### **Time-series foundation models**

Examples:

* Generic forecasting transformers  
* Time-series tokenization frameworks

**Finding:** Large-scale pretraining creates transferable temporal representations.

**Limitation:** They model sequences of numbers, not rich customer-event ecosystems.

---

## **6\. nuFormer (Braithwaite et al., 2025\)**

### **Approach**

Closest precursor to PRAGMA.

* Customer represented through transaction ledger events.  
* Learns user embeddings from spending behavior.  
* Uses transformer-style sequence modeling.

### **Key finding**

Large-scale transaction histories can be pretrained and transferred to recommendation tasks.

### **Limitation identified by PRAGMA**

* Primarily one event stream (transactions).  
* Little explicit modeling of static customer state.  
* Evaluated mainly on recommendation use cases.

So it answers:

> "What product should we recommend?"

but not necessarily

> "Will this customer churn?"

> "Is this transaction fraudulent?"

> "Will this customer need support?"

> "What is the risk level?"

with one shared backbone.

---

## **7\. TransactionGPT (Dou et al., 2025\)**

### **Approach**

Treats transaction histories similarly to language.

* Transactions become tokens.  
* Generative objective predicts future trajectories.  
* Supports anomaly detection and sequence generation.

### **Key finding**

Banking ledgers exhibit language-like structure that can be modeled with LLM-style pretraining.

### **Limitation identified by PRAGMA**

* Transaction stream remains the dominant modality.  
* Limited incorporation of profile/state information.  
* Evaluation centered on:  
  * anomaly detection  
  * transaction generation  
  * trajectory modeling

rather than a broad suite of discriminative banking tasks.

---

## **The progression PRAGMA is arguing**

You can think of the literature as evolving through four stages:

| Stage | Representative papers | What is modeled |
| ----- | ----- | ----- |
| Tabular foundation | TabTransformer, FT-Transformer | Static customer row |
| Sequential behavior | SASRec, BERT4Rec | Event sequence |
| Financial foundation models | FinBERT, time-series FMs | Text or temporal signals |
| Banking ledger models | nuFormer, TransactionGPT | Transaction histories |

PRAGMA's claimed contribution is the **next step**:

Static Profile State  
        \+  
Multiple Event Streams  
        \+  
Shared Transformer Backbone  
        \+  
Transfer Across Many Banking Tasks

So the novelty PRAGMA is positioning is not merely "using transformers for transactions," but building a **unified customer foundation model** that jointly encodes **who the customer is (profile state)** and **what the customer does across multiple banking channels (multi-source events)**, then reuses that representation across many predictive tasks.

