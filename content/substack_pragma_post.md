If you've worked in applied ML in the FS/Fintech world, you know the hassle of data wrangling and feature engineering. Before AutoML entered the picture, production ML was fragmented and tedious. Fast forward to today, and ML models are a staple cog not just in fintechs but in banks and NBFCs too. I believe LLMs have opened a new era of AI/ML implementation — and it's spilling well past language.

Last month I came across Revolut's paper on **PRAGMA** (Profile and Event Representations for Banking), and it struck me as a new paradigm. They built a foundation model that understands a user by looking at two things: their static profile (age, account type) and their sequence of transactions (coffee, then groceries, then a bill payment). It performed strongly across several banking tasks.

A little research and I was down the rabbit hole of **Tabular Large Models** — a category of AI seeking its own ChatGPT moment. Think of every "AI financial advisor" app that promises to understand your money but ends up giving you generic advice instead of actually reading your transaction history. That gap is exactly what tabular foundation models are trying to close.

And that rabbit hole got a very concrete data point just as I was writing this: **SAP has agreed to invest over €1 billion over four years to acquire Prior Labs**, the team behind TabPFN — a foundation model built specifically for tabular, structured data. That's not a research-paper flex. That's boardroom capital, betting that the "read a spreadsheet the way we read language" idea is close to an inflection point.

So I wanted to know: could I build something like PRAGMA myself? And more importantly, could I make it work on an open-source dataset so other people could pull it apart, question it, and play with it?

Here's exactly how I broke down the math, built the architecture from scratch, and benchmarked it against industry standards using the public IEEE-CIS Fraud Detection dataset.

---

## The Historical Context: Why This Is Harder Than It Looks

Several data leaders in BFSI and fintech are still dismissive that transformer architectures can surpass their existing, heavily-tuned traditional ML setups. Even fintech CXOs and ML leaders at established players haven't fully priced in the leap large models are attempting here — and there's good reason for the skepticism.

The clearest real-world proof point is **Nubank**, which acquired the AI startup Hyperplane in 2024 specifically to accelerate this bet. The combined team built billion-parameter transformer models — internally called nuFormer — that treat a customer's transaction history the way a language model treats a sentence: the same mechanism that predicts the next word in a sentence can be pointed at predicting the next financial behavior in a spending history. It wasn't a trivial win. Getting transformer-based models to a point where they reliably beat tree-based baselines took real engineering effort — new sequence pipelines, GPU infrastructure that their old tree models never needed, and an entirely different training paradigm (tuning transformer weights, not fitting boosted trees). The payoff, once it landed: a genuine **1.20% average AUC lift across benchmark tasks**, deployed into production decision engines serving over 100 million customers.

That's the honest baseline expectation for anyone attempting this: transformers on tabular financial data are not a free win. Beating a heavily optimized XGBoost model is genuinely hard, and it's exactly what makes Revolut's PRAGMA architecture worth taking seriously — they published a specific recipe for how to close that gap.

---

## The Core Idea: Two Streams of Information

When I read the paper, the architecture made a lot of sense. You can't just throw raw transaction logs into a standard model and expect it to understand a user's broader context. PRAGMA splits the problem into two distinct encoders, then fuses them together. Here's the math and mechanics behind both, in plain terms.

### 1. The Profile Encoder (Understanding the User)

The profile encoder handles the slow-changing stuff. In my dataset, that meant things like the user's email domain (`P_emaildomain`), card network (`card4`), and numerical features like time since their first transaction.

You can't feed a category like `"gmail.com"` directly into a neural network — it needs to become a number first. The standard trick is a **learned embedding**: every category gets mapped to a small dense vector, and the network learns what that vector should be during training.

**Concrete example:** say our dataset has three possible email domains: `yahoo.com`, `gmail.com`, `hotmail.com`. `"gmail.com"` starts life as a one-hot vector — `[0, 1, 0]`, a 1 in its own slot and 0s everywhere else. That's just a placeholder; it carries no meaning about which email providers are similar to each other.

During training, the model learns an **embedding matrix** — think of it as a lookup table with one row per category and a handful of learned numbers per row. Multiplying that one-hot vector by the embedding matrix simply "plucks out" the row for `gmail.com`, turning it into a dense vector like `[-0.45, 3.14, 0.42, -0.99]`. Do this for every categorical feature (email domain, card type, and so on), then concatenate all those vectors together with the raw numeric features (like account age) into one input vector.

That combined vector is passed through a standard **Multi-Layer Perceptron (MLP)** — a couple of layers of linear transformations with a non-linearity (ReLU) in between. The output is a single dense vector representing the user's baseline profile state.

### 2. The Event Encoder (Understanding the Behavior)

This was the fun part. The event encoder treats a user's transaction history like a sentence: if I buy a flight, book a hotel, and then there's a charge at a petrol pump a thousand kilometres away, the *order* of those events matters.

To handle this, I used a **Multi-Head Transformer Encoder** — the same core mechanism behind ChatGPT, repurposed for transactions instead of words.

First, every transaction gets converted into a vector, the same embedding trick as above. But transformers don't natively understand sequence order — left to itself, the model just sees a bag of transactions. So I added **positional encodings**: a set of mathematical patterns (built from sines and cosines) added to each transaction's vector that act like a timestamp, letting the model tell that a Starbucks purchase happened before an Amazon order.

**Concrete example:** a user has three recent transactions — $15 at Starbucks, $120 at Amazon, $50 at Shell. Each becomes a vector. Add the positional pattern for "1st," "2nd," and "3rd" respectively, and the model now knows the order they happened in, not just that they happened.

Then comes **self-attention** — the actual engine of the transformer. For every transaction, the model asks: "how relevant is every other transaction in this user's history to understanding this one?" It does this by projecting the input into three learned views — Queries, Keys, and Values — and computing a weighted average of Values, where the weights come from how well each transaction's Query matches every other transaction's Key.

One practical wrinkle: users don't all have the same number of transactions — some have 10, some have 1,000. I padded every sequence to a fixed length so they could be batched together, but I didn't want the model paying attention to those empty padded slots. So the attention calculation includes a **mask** that forces the model to ignore padding entirely.

After a few transformer layers, I pooled the output into a single dense vector representing the user's recent behavior. Fuse that with the profile vector from step one, and you have PRAGMA's full representation of a customer — who they are, and what they've been doing.

![architecture diagram][image1]

---

## How Does It Actually Perform?

Revolut evaluated PRAGMA across several credit and risk tasks on proprietary data. I don't have their data, so I adapted the open IEEE-CIS Fraud Detection dataset to a single, well-defined task — fraud detection — and split it strictly by time, so the model can never see the future during training.

I benchmarked the fine-tuned PRAGMA model against the classical tabular models that actually run in most production fraud stacks today. Here's the real result, on the held-out test set, measured on AUC-ROC:

| Rank | Model | Test AUC-ROC | Notes |
|---|---|---|---|
| 1 | XGBoost | 0.8880 | Tree-based models remain the gold standard for tabular data. |
| 2 | LightGBM | 0.8873 | Highly efficient; near-identical to XGBoost. |
| 3 | CatBoost | 0.8716 | Strong baseline, excellent native categorical handling. |
| 4 | Random Forest | 0.8532 | Solid baseline ensemble. |
| 5 | **PRAGMA** | **0.7939** | My transformer replication. |
| 6 | Logistic Regression | 0.7568 | Linear baseline. |

I'm not going to dress that up: PRAGMA finished behind every tree-based model. I could have kept iterating until I found a setup where it "won," but that's exactly the kind of result-shopping that makes so much published ML hard to trust. This is the number I got, on a fair, time-based split, and it's more useful published honestly than quietly replaced.

### What These Numbers Tell Me

**The tabular data advantage is real.** XGBoost and LightGBM dominate tabular datasets like IEEE-CIS almost out of the box. Deep learning architectures like PRAGMA typically need a lot more scale before they overtake trees on structured data — this replication is a clean illustration of that, not an exception to it.

**Scale is the missing variable, not the architecture.** I trained PRAGMA for 5 epochs at an embedding dimension of 64, on hardware I could actually get my hands on. Revolut trains PRAGMA on billions of proprietary transactions across GPU clusters most teams will never budget for. The transformer's real advantage — reasoning over long, messy behavioral history — is a scale-dependent one, and this run simply wasn't run at that scale.

**The replication still succeeded on its own terms.** Despite losing to XGBoost, a score of 0.7939 proves the PRAGMA architecture *works*: it learned meaningful structure directly from raw transaction streams, with zero manual feature engineering, and comfortably beat a logistic regression baseline. That's not nothing — it's a foundation model architecture standing on its own two feet, just not yet at tree-model altitude.

The natural next question — does self-supervised pretraining meaningfully lift PRAGMA over training it from scratch, and does the profile stream actually add anything the event stream doesn't already capture — is exactly what the next dispatch in this series will answer, with proper ablations and multiple seeds rather than a single run.

## Industry Validation: Plaid's Sequential Foundation Model

Interestingly, just as I was preparing to publish this, [Plaid released a blog post](https://plaid.com/blog/sequential-foundation-model/) detailing their own research on "sequential foundation models" for tabular financial data.

Plaid's engineering team arrived at the same conclusion as Revolut, Nubank, and this replication project: financial data is fundamentally a language. They treat sequences of user transactions — ACH transfers, payroll, subscriptions — as a "grammar of money," pretraining once on a large, unlabeled corpus of financial activity and adapting that shared model to specific tasks with a small amount of labeled data. The results, once deployed, are not subtle: **26.5% more dollar value in returns prevented at a fixed 1% action rate** for ACH payment risk, and a **13.6% reduction in default risk at a 70% approval rate** for credit underwriting.

That's the pattern across every serious attempt at this so far: the entire industry is pivoting away from static, point-in-time tabular features — where XGBoost has historically reigned — toward treating financial ledgers as temporal sequences, where transformers have room to shine. Plaid and Nubank are running these architectures on billions of proprietary transactions; this replication shows the same underlying idea holds even when adapted to a public, open-source dataset — even before it's been given the scale to fully prove itself.

## Wrapping Up

Building this was a genuine learning experience. It proved you don't need a billion-dollar data warehouse to experiment with the same ideas behind state-of-the-art financial foundation models — open data, a pseudo-ClientID, and doing the math by hand gets you a working pipeline that mirrors Revolut's core claims, even if it hasn't yet matched their results.

We are still in the early days of large models for tabular, structured data — roughly where language models were before scale changed everything. Traditional tree-based models are excellent at squeezing signal out of hand-engineered features, but they largely ignore the raw, sequential, unstructured story sitting underneath those features — and they demand a lot of manual effort to keep working as new features get engineered by hand, task by task. Foundation models like PRAGMA start from a harder position but a more scalable one: less manual effort per task, more of the signal captured directly from raw data. Closing the current performance gap isn't a question of *if* so much as *at what scale* — and that's precisely the gap this series is going to keep measuring, one honest experiment at a time.

If you want to poke around the code, I've open-sourced the whole PyTorch pipeline — you can pull it down and run it on a Google Colab notebook today.

---

## References & Further Reading

For those newer to the underlying mechanics, here's a quick glossary of the concepts used in this architecture, and *why* each one is necessary:

- **One-hot vector:** a way to represent a category (like a word or a category label) as a list of 0s with a single 1. Three categories — Apple, Banana, Orange — and "Banana" becomes `[0, 1, 0]`.
  - *Why we use it:* neural networks only understand numbers. One-hot encoding is the simplest way to turn a category into something a network can multiply.
- **Embedding matrix:** a lookup table the network learns during training, converting rigid one-hot vectors into dense, meaningful vectors.
  - *Why we use it:* a one-hot vector for `"gmail.com"` has no inherent relationship to `"yahoo.com"`. An embedding matrix lets the network learn that relationship — perhaps discovering that mainstream email domains cluster together, while a disposable "burner" domain sits far away in that space.
- **Multi-Layer Perceptron (MLP):** the classic neural network — layers of neurons, each connected to every neuron in the next layer.
  - *Why we use it:* it's excellent at finding non-linear patterns in static, tabular data, like a user's profile.
- **Transformer & self-attention:** the architecture behind ChatGPT, built around "self-attention" for processing sequences.
  - *Why we use it:* to understand a sequence of transactions, the model needs to weigh how a $5,000 transfer relates to a $2 coffee purchase the next day. Self-attention is exactly the mechanism for weighing the relevance of different events in a timeline against one another.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnAAAACbCAMAAADV0Vb5AAADAFBMVEUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACzMPSIAAAAj3RSTlMAAQIDBAUGBwgJCgsMDg8REhMUFRYXGRobHR4fICUoKSosMjM8PkBBREVGR0hKS05PUVVWXF1gYmNkZmxtb3N0d3h5ent8foCBgoaIioyNjpCUlZaXnZ6foKGkp6mrr7K0tba3ubvBwsTFxsnMztDR0tXW2Nrb3N7g4eLj5Ofo6ers7e7w8fLz9vj5+vv9/g9IUGAAAAKrSURBVHhe7dy9ahRhGIbhiYpokTYINiktPQFB0E7PwTJ1UBC1MgjBRnvxFASFWApqYyXYiL1oKYKCFoI/kAi7Lwm732TzZLNzXVX2gUBmuZmPkMl2HQAAAAAAAAAAAAAwv5bq0HUP6zCN9ZGvb62MvJjOq6cjL3r9AHNh9F04ulcxQ6NvyLZjdYCDJDiiBEeU4IgSHFGCI0pwRAmOKMERJTiiBEeU4IgSHFGCI0pwRJ2ow27PMDXarAP85w5HlOCIEhxRgiNKcEQJjijBESU4ogRHlOCIEhxRgiNKcEQJjijBESU4ogRH1C5P/PZ288zP23WDMbO8w93v9MYEswwOJup1pF66Wpcde3yQ8stndWGo+gS3sVyXCS5euFEnBqrPkdraW9cdrwND1Sc46E1wRAmOKMERJTiiBEeU4IgSHFGCI0pwRAmOqD7B/ajDRL/rwFD1Ce7Oi7pM8OZ6XRiqPo8ndVtbdYHp9LnDQW+CI0pwRAmOKMERJTiiBEeU4IgSHFGCI0pwRAmOKMERJTiiBEeU4IgSHFG9nvjdw9q5L/fqBmNmeYd73j2pE4ybZXCfug91gnFtR+qptdU6jdnjQ6W3fX70rU4MTtsdbnO1Lg3O3q0Lw9MW3D6t1IHBaQruZB0ana4Dg9MUHOxX2y8NGY/rcCQtxlXM3DwG974OR9JiXMXMOVKJEhxRgiNKcEQJjijBEdUU3J86NNrv93P0NQX3qw6NPtaBwWkKDgAAAAAAAA7PUh0W2vJGt163hXX+2jxe7LD+lvq9Dovs3Vw+nTOP/7V1gN6+rssC+/qgLnNgUY/Uf4dnizk8e6Z3+UpdJjjMqx3WkcqhExxRgiNKcEQJjijBESU4ogRHlOCIEhxRgiNKcEQJjijBESU4AAAAAAAAAAAAAAB2/AUdMy5zxPzm/wAAAABJRU5ErkJggg==>

