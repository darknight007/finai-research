# Outlier Alpha Content & Research Roadmap (2026)

Follow-up plan built from a 10-project AI-x-fraud research brainstorm, filtered for solo/Colab feasibility, sequenced into a Substack posting cadence, and pointed at an eventual research paper.

## Feasibility pass on the 10 candidate projects

| # | Project | Verdict | Why |
|---|---|---|---|
| 3 | Tabular FMs (TabTransformer/FT-Transformer/TabPFN) | **Build next, immediately** | Already promised in the PRAGMA Substack post's closing line; reuses the existing pipeline/splits. Zero new infra. |
| 1 | Graph Neural Networks for fraud | **Build second** | IEEE-CIS + Kaggle Credit Card Fraud + Elliptic Bitcoin are all genuinely good public graph testbeds. Elliptic is graph-native ground truth — real technical credibility. |
| 9 | Explainable Fraud AI (SHAP/LLM/Graph) | **Cheap add-on, anytime** | No new training, just analysis on models already built. Strengthens any eventual paper's methods section. |
| 5 | Synthetic Fraud Generation (CTGAN/TimeGAN) | **Build, mid-series** | Tractable, well-supported libraries. Enables a callback question: does synthetic augmentation close PRAGMA's gap to XGBoost? |
| 4 | FraudRAG | **Build, but scope tightly** | Visa dispute manuals and RBI circulars are genuinely public — buildable. No canonical paper exists yet, which is the actual novelty claim. |
| 2 | Agentic Fraud Investigation | **Build, but fix the eval first** | LangGraph/AutoGen prototypes are easy; a rigorous comparison against "human analyst" needs a small labeled eval rubric (even 50-100 synthetic cases) before building agents, or it's a demo, not research. |
| 8 | Autonomous Fraud Ops Center | **Fold into Project 2's finale** | It's Project 2 taken to its limit, not a separate project. |
| 7 | Temporal Transformers (Informer/PatchTST) | **Fold into Project 3 as an ablation** | Built for continuous time-series forecasting; irregular discrete transaction sequences are already handled by PRAGMA's event encoder. Standalone it's a thin contribution; as a sequence-model ablation column inside Project 3 it's valuable. |
| 6 | Behavioural Biometrics | **Survey/commentary post only** | No solid public dataset ties keystroke/touch dynamics to actual fraud labels at retail-banking scale. Cover as industry commentary (Mastercard/BioCatch), honest about the reproduction limitation. |
| 10 | PRAGMA for Fraud (PRAGMA + Graph + RAG) | **The capstone — and the paper** | The one item nobody has published. Everything else on the list is reproduction-with-a-twist; this is a genuine synthesis contribution. |

## Recommended posting sequence (through end of 2026)

Structured so each post reuses the previous post's infrastructure — that's what makes weekly/biweekly cadence survivable solo, and it's also what makes the eventual paper's related-work section write itself.

**Aug — Finish what PRAGMA started** (closes the promise made in post #1, cheapest lift)
1. DL baselines under identical budgets — TabTransformer, FT-Transformer vs PRAGMA vs trees
2. Ablations with seeds + confidence intervals — profile-only vs events-only vs full, with/without pretraining
3. Synthesis: *"Do Foundation Models Replace Feature Engineering in Banking?"* — ties back to the SAP/TabPFN hook from post 1

**Sept–Oct — Graph intelligence arc**
4. Why rule engines and trees miss fraud rings (lit review: GraphSAGE, EvolveGCN, Visa/Mastercard/PayPal production notes)
5. Build the entity graph (customer→card→merchant→device→IP) on IEEE-CIS, benchmark GraphSAGE/GAT vs XGBoost/LightGBM
6. Add Elliptic Bitcoin as a second, graph-native testbed — stronger validation than IEEE-CIS alone
7. Synthesis: *"Can Graph AI Reduce False Positives in Credit Card Fraud?"*

**Nov — Cheap high-value connective posts**
8. SHAP vs LLM explanation vs Graph explanation, on everything built so far
9. Synthetic fraud generation (CTGAN/TimeGAN) — does it close PRAGMA's gap to XGBoost?

**Dec onward — The original-research arc (this becomes the paper)**
10. Multi-agent fraud investigation prototype, benchmarked against a real labeled eval set (build the rubric before the agents)
11. FraudRAG — public policy/dispute-manual retrieval grounding investigation writeups
12. Capstone: **PRAGMA + Graph + RAG**, benchmarked end-to-end: rules → XGBoost → Graph → PRAGMA → PRAGMA+GraphRAG

~12 posts, matching a realistic weekly-to-biweekly cadence for the rest of the year, ending exactly where the paper should start.

## On the actual paper

Don't publish "we reproduced GraphSAGE" or "we reproduced PRAGMA" alone — neither clears the bar for original research; both are exactly the reproduction the roadmap warns against. The capstone (post #12) is the real contribution: nobody has published the fusion of profile+event foundation models, graph-structured entity relationships, and retrieval-augmented reasoning for fraud investigation, with a full ablation ladder against production-realistic baselines. Posts 1-11 double as pilot results and related-work groundwork before approaching a collaborator.

**Venue:** aim for **ACM ICAIF** (International Conference on AI in Finance) — the most directly-fitting venue for benchmarked AI×finance work, realistically reachable within a year. The **NeurIPS Table Representation Learning workshop** is a good secondary target if leading with the tabular-FM angle instead. A financial-NLP workshop at ACL/EMNLP fits if the agentic/RAG angle ends up being the stronger half.

**Professor collaboration:** bring someone in once posts 1-9 exist (the pilot evidence), not before — a professor can evaluate a working empirical pipeline far better than a proposal. Look for faculty in one of two lanes depending on which half of the capstone leads: tabular/graph deep learning for structured data, or NLP agents/retrieval-augmented reasoning applied to finance. A co-author matters less for the coding and more for statistical rigor on the ablations, and for the credibility a university affiliation adds when pitching to BFSI audiences.
