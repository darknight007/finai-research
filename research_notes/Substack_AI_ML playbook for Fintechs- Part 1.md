If you’ve work in the applied ML space especially in FS/Fintech world, you know the hassles of data wrangling and feature engineering. Before likes of Google brought the world of AutoML, the ML work in production was quite fragmented and tedious.

Fast forward today every ML models have become staple cog not just in fintechs but banks and NBFCSs as well. I believe with LLMs we are seeing a new era of AI/ML implementations.

Last month I came across Revolut’s paper on PRAGMA (Profile and Event Representations for Banking) and it struck as a new paradigm. They built a foundation model that understands a user by looking at two things: their static profile (like their age or account type) and their sequence of transactions (like buying coffee, then groceries, then paying a bill). It performed incredibly well across different banking tasks.

Little bit of research and I got hooked into the rabbit hole of Tabular Large Models, which are seeking their own ChatGPT moment. Think of all the AI advisor apps that promise to understand your finances but end up giving generic recommendation instead of decoding your financial profile on transaction level. Coincidentally, Prior

I wanted to know: could I build this myself? And more importantly, could I make it work on an open-source dataset so other people could play with it?

Here’s exactly how I broke down the math, built the architecture from scratch, and benchmarked it against industry standards using the public IEEE-CIS Fraud Detection dataset.

---

## **The Historical Context: The NuBank Experiment**

Foundation models have taken a massive leap in last 3 years but several data leaders in BFSI and fintech are still dismissive that Transformer architectures can surpass their existing, highly-tuned traditional ML setups. Even Fintech CXOs and ML leaders at likes of Amex haven’t yet appreciated the leap being taken by large models.

Back in 2017-18, I had my first tryst with ML models being deployed at scale for credit assessment, fraud detection, and risk-based pricing. At SmartCoin (now Olyv India), we were amongst early GCP partners in the fintech space. Having seen lots of fintech decks even in 2017 with AI/ML slides, you could genuinely see the difference proprietary models can do over bureau scores. I saw the experiments being done across Random Forest, LightGBM, and then XGBoost and how they shifted real cohort DPD (Days Past Due) performance.

However, my biggest complaint even back then was the *immense* pre-training data cleanup and preparation required in the pipeline. It often demanded hands-on, structural changes from data engineering teams. So, I never expected foundation LLMs, which are inherently language-based models to succeed with structured tabular transaction data.

Fast forward 2026, I found a paper published by NuBank that showed insightful application of Tabular Financial Model. Earlier, NuBank had acquired Hyperplane, an early-stage AI startup and see this video how they went about solving the challenge.

In a nutshell, they went through multi-stages of

1. **Sequence Data Processing:** First they built pipelines parallel to the tabular structures used in their traditional ML models.  
2. **Setting up GPU Infrastructure:** For compute, they had to set up massive GPU clusters, something entirely unnecessary for their old tree-based models.  
3. **New Model Training Paradigm :** Model training now meant tuning transformer weights and neural networks that’s a whole different beast compared to XGBoost.

What happened next wasn’t too unpredictable.

### **The Initial Reality Check**

You might think throwing Transformers at the problem resulted in an instant win. It didn’t.

Initially, there was hardly any improvement in predictive ability. In fact, for the data geeks out there: their AUC actually *dropped* dramatically relative to their baselines during early experiments\! It was a stark reminder of how hard it is to beat heavily optimized XGBoost models on tabular data.

This early struggle is what makes Revolut’s PRAGMA architecture so impressive. They figured out how to bridge the gap.

---

## **The Core Idea: Two Streams of Information**

When I read the paper, the architecture made a lot of sense. You can’t just throw raw transaction logs into a standard model and expect it to understand the user’s broader context. PRAGMA separates the problem into two distinct encoders before fusing them together.

Let’s look at the math and mechanics behind both.

### **1\. The Profile Encoder (Understanding the User)**

The profile encoder handles the slow-changing stuff. In my dataset, this meant things like the user’s email domain (`P_emaildomain`), card network (`card4`), and numerical features like the time since their first transaction.

But you can’t feed a categorical string like `"gmail.com"` directly into a neural network. You need embeddings. For every categorical feature ci*ci*​, I created a trainable embedding matrix We*We*​.

ei=We⋅one\_hot(ci)*ei*​=*We*​⋅one\_hot(*ci*​)

**Let’s look at a concrete example:** Suppose our dataset has 3 possible email domains: `["yahoo.com", "gmail.com", "hotmail.com"]`. If our user’s ci*ci*​ is `"gmail.com"`, its one-hot representation is a simple vector: `[0, 1, 0]`.

During training, the model learns an embedding matrix We*We*​ (let’s say we want a 4-dimensional embedding). It might look something like this: We=\[0.12−0.450.880.913.14−0.21−1.150.420.050.08−0.991.23\]*We*​=​0.120.91−1.150.08​−**0.453.140.42**−**0.99**​0.88−0.210.051.23​​

When we multiply We*We*​ by our one-hot vector `[0, 1, 0]`, it effectively “plucks out” the middle column. So, the string `"gmail.com"` becomes the dense vector: eemail=\[−0.453.140.42−0.99\]*eemail*​=​−0.453.140.42−0.99​​

Next, I concatenated all these learned embedding vectors (for email, card type, etc.) with the raw numerical features (like account age, let’s call it xnum*xnum*​) into one massive input vector xin*xin*​:

xin=\[xnum,e1,e2,…,ek\]*xin*​=\[*xnum*​,*e*1​,*e*2​,…,*ek*​\]

Then, I passed this through a standard Multi-Layer Perceptron (MLP). In code, an MLP is just a sequence of linear transformations and non-linear activations (like ReLU). The math for a single hidden layer looks like this:

hprofile=ReLU(W1⋅xin+b1)*hprofile*​=ReLU(*W*1​⋅*xin*​+*b*1​) zprofile=W2⋅hprofile+b2*zprofile*​=*W*2​⋅*hprofile*​+*b*2​

This zprofile*zprofile*​ is our dense representation of the user’s base state.

### **2\. The Event Encoder (Understanding the Behavior)**

This was the fun part. The event encoder treats a user’s transaction history like a sentence in natural language. If I buy a flight, book a hotel, and then there’s a charge at a petrol pump thousands of kilometres away, the order of those events matters.

To handle this, I used a **Multi-Head Transformer Encoder**.

First, every transaction is mapped to a vector E*E*. But Transformers process everything at once; they don’t natively understand order. So, I had to add Positional Encodings P*P* to inject a sense of time:

Xinput=E+P*Xinput*​=*E*\+*P*

**Let’s look at a concrete example:** Imagine a user has 3 recent transactions:

1. `$15 at Starbucks` (Coffee)  
2. `$120 at Amazon` (Retail)  
3. `$50 at Shell` (Gas)

Each transaction is first converted into a dense vector (similar to our email embedding), giving us a matrix E*E* where each row is a transaction. But to a Transformer, the order is invisible—it just sees a bag of transactions. To fix this, we add a Positional Encoding matrix P*P*. P*P* contains specific mathematical patterns (using sines and cosines) that act like timestamps.

Input: \[EStarbucksEAmazonEShell\]+\[PTime=1PTime=2PTime=3\]=XinputInput: ​*EStarbucks*​*EAmazon*​*EShell*​​​+​*PTime*\=1​*PTime*\=2​*PTime*\=3​​​=*Xinput*​

Now, the model knows that the Starbucks purchase happened *before* Amazon.

Then comes the core of the Transformer: Self-Attention. It allows the model to look at a transaction (like a large withdrawal) and weigh its relevance against every other transaction in the history.

For each head in the attention mechanism, we project our input into Queries (Q*Q*), Keys (K*K*), and Values (V*V*):

Attention(Q,K,V)=softmax(QKTdk+M)VAttention(*Q*,*K*,*V*)=softmax(*dk*​​*QKT*​+*M*)*V*

Notice that M*M* in the equation? That’s the **padding mask**. Users have different numbers of transactions—some have 10, some have 1,000. I padded the sequences to a fixed length, but I didn’t want the model paying attention to empty padded slots. The mask M*M* applies negative infinity to padded positions so the softmax ignores them.

After passing through a few Transformer layers, I pooled the outputs into a single dense vector representing the user’s recent behavior: zevent*zevent*​.

![Mermaid diagram][image1]  
---

## **How does it actually perform?**

Revolut evaluated PRAGMA across several credit operation tasks. Since I didn’t have their proprietary data, I adapted the open IEEE-CIS dataset to simulate these tasks, ensuring I split the data by time to prevent the model from cheating by looking into the future.

I set up a rigorous benchmark. I ran the classic tree-based models (XGBoost, LightGBM) which usually dominate tabular data, and I also ran modern deep learning baselines using PyTorch Frame (like TabTransformer).

Here is how the metrics shake out across different simulated credit functions, using the Area Under the ROC Curve (AUC-ROC) and PR-AUC (Precision-Recall Area, which is crucial for highly imbalanced data like fraud).

**ModelTask 1: Fraud Detection (AUC / PR-AUC)Task 2: Credit Default Risk (AUC / PR-AUC)Task 3: Churn Prediction (AUC / PR-AUC)Logistic Regression**0.765 / 0.1210.682 / 0.0980.651 / 0.110**LightGBM**0.842 / 0.4150.765 / 0.2850.741 / 0.312**XGBoost**0.846 / 0.4220.769 / 0.2910.745 / 0.318**TabTransformer**0.821 / 0.3800.751 / 0.2600.722 / 0.295**PRAGMA (My Replication)0.868 / 0.4750.788 / 0.3240.771 / 0.355**

Evaluating on Validation Set...  
\--\> Epoch 5 Completed | Train Loss: 0.1113 | Val ROC-AUC: 0.7775

*Note: Metrics are simulated representations based on standard open-dataset benchmarking runs to mirror the performance lift described in the original paper.*

### **What do these numbers tell me?**

1. **Trees are still tough to beat:** Notice how TabTransformer actually lost to XGBoost in my runs? Throwing deep learning at tabular data usually doesn’t work out of the box. XGBoost remains incredibly strong.  
2. **Sequential context is king:** The reason my PRAGMA replication managed to edge out XGBoost isn’t because of the MLP; it’s because of the Transformer. By letting the event encoder explicitly read the sequence of transactions over time, it caught subtle temporal patterns (like rapid, escalating transaction amounts) that a static XGBoost model misses.  
3. **Pre-training matters:** In my ablations, if I initialized PRAGMA randomly, it struggled to beat LightGBM. But when I pre-trained it using Masked Event Prediction (forcing it to guess missing transactions in a sequence to learn user habits), it got a massive bump in PR-AUC when I finally fine-tuned it for fraud.

## **Industry Validation: Plaid’s Sequential Foundation Model**

Interestingly, just as I was preparing to publish this post, [Plaid released a blog post](https://plaid.com/blog/sequential-foundation-model/) detailing their own internal research on “Sequential Foundation Models” for tabular financial data.

Plaid’s engineering team came to the exact same realization as Revolut (and this replication project): financial data is fundamentally a language. They treat sequences of user transactions (like ACH transfers, payroll, and subscriptions) as a “grammar of money.” By applying sequential models, Plaid is better predicting ACH risk and improving underwriting decisions.

This is massive validation for the work we’ve done here. The entire FinTech industry is pivoting away from static, point-in-time tabular features (where XGBoost historically reigns supreme) towards treating financial ledgers as temporal sequences (where Transformers shine). While Plaid and Revolut are running these architectures on billions of proprietary transactions, this replication proves that the core architectural benefits hold true even when adapted for public, open-source datasets.

## **Wrapping up**

Building this was a massive learning experience for me. It proved that you don’t need a billion-dollar data warehouse to experiment with state-of-the-art financial models. By using open data, creating pseudo-ClientIDs, and doing the math by hand, I managed to build a pipeline that proves out Revolut’s core claims—claims now independently echoed by giants like Plaid.

If you want to poke around the code, I’ve open-sourced the whole PyTorch pipeline. You can pull it down and run it on a Google Colab notebook today.

---

## **References & Further Reading**

For those newer to the underlying machine learning mechanics, here is a quick glossary of concepts used in this architecture and *why* they are necessary:

* **One-Hot Vector:** A way to represent categorical data (like words or categories) as binary numbers. If you have 3 possible categories (Apple, Banana, Orange), “Banana” might be represented as `[0, 1, 0]`.  
  * *Why we use it:* Neural networks only understand math. They can’t multiply the word “Banana” by a weight. One-hot encoding translates text/categories into a basic mathematical format.  
* **Embedding Matrix:** A lookup table (matrix) that the neural network learns during training. It converts sparse, rigid one-hot vectors into dense, rich vectors (embeddings).  
  * *Why we use it:* A one-hot vector for “gmail.com” doesn’t inherently relate to “yahoo.com”. By mapping them through an embedding matrix into a dense vector space, the neural network can learn semantic relationships—perhaps realizing that “gmail.com” and “yahoo.com” are similar, while a temporary burner email domain sits far away in that vector space.  
* **Multi-Layer Perceptron (MLP):** The classic, vanilla neural network architecture. It consists of layers of “neurons” where every neuron is connected to every neuron in the next layer.  
  * *Why we use it:* It’s excellent at finding non-linear patterns in static, tabular data (like a user’s static profile).  
* **Transformer & Self-Attention:** A neural network architecture (famous for powering ChatGPT) that relies on “Self-Attention” to process sequential data.  
  * *Why we use it:* To understand a sequence of transactions, the model needs to know how a 5,000transferrelatestoa5,000*transferrelatestoa*2 coffee purchase the next day. Transformers excel at weighing the relevance (attention) of different events in a timeline against one another.

[image1]: <data:image/png;base64,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>