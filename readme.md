# Chapter 13: SFT vs. RAG — When to Bake Knowledge In vs. Retrieve It

> *A chapter from* **Design of Agentic Systems with Case Studies**
> *INFO 7375 — Prompt Engineering for Generative AI, Northeastern University*

---

## Overview

This repository contains the complete submission for Chapter 13 of the course book. The chapter makes a single architectural argument: the decision between Supervised Fine-Tuning and Retrieval-Augmented Generation is not a preference or a latency tradeoff. It is a routing problem driven by knowledge volatility. Route by intuition and you ship a system that fails silently — after deployment, in production, in front of a client.

The chapter opens with a forensic reconstruction of a real compliance failure, develops a quantitative decision framework built around the knowledge volatility index V, and closes with two case studies at opposite ends of the volatility spectrum: a law firm deploying a legal research assistant on stable statutory knowledge, and the 80 Days to Stay visa search system operating on continuously changing SEC Form D data.

---

## Repository Structure

```
.
├── chapter13.md              # Full chapter prose (publication-ready)
├── notebook/
│   └── chapter13_demo.ipynb  # Runnable demo — Mistral-7B on Kaggle
├── figures/
│   └── figures.html          # All four publication figures (export file)
├── authors_note.md           # 3-page Author's Note
└── README.md
```

---

## The Architectural Argument

The chapter demonstrates that SFT and RAG are not competing alternatives but complementary strategies whose selection is determined by a single measurable quantity:

$$V = \frac{1}{T_{\text{update}}}$$

where $T_{\text{update}}$ is the expected interval between updates to a knowledge category, expressed as a multiple of the model retraining cycle. High-V knowledge belongs in retrieval. Low-V knowledge belongs in weights. The routing decision follows from the number.

The chapter traces a five-link causal chain from this architectural decision to a client deficiency notice, demonstrates the failure in code, and shows what the monitoring infrastructure must look like to catch it before it materializes.

---

## Demo Notebook

The notebook runs on **Kaggle's free T4 GPU**. No API key required.

### Setup

1. Go to [kaggle.com](https://www.kaggle.com) and create a new notebook
2. Settings → Accelerator → **GPU T4 x2**
3. Settings → Internet → **On**
4. Paste the contents of `notebook/chapter13_demo.ipynb` and click **Run All**
5. First run downloads Mistral-7B-Instruct-v0.2 (~4 minutes). Subsequent runs are instant from cache.

### What the Demo Shows

| Part | What it demonstrates |
|---|---|
| Part 1 | SFT mock backbone frozen at Nov 2023 — returns $15,000 |
| Part 2 | RAG pipeline over live document store — returns $25,000 |
| Part 3 | Router classifying HIGH-V vs LOW-V with confidence and reasoning |
| Part 5 | Deliberate failure trigger: forced SFT on volatile query reproduces the Chicago error |
| Part 6 | Cache hit/miss demo + cache TTL failure trigger |
| Part 7 | Monitoring scaffold: alert fires when SFT returns $15,000 against ground truth of $25,000 |

### Expected Output

**Forced SFT (the failure):**
```
source: parametric_memory — frozen Nov 2023
effective_route: SFT
ANSWER: The SEC penalty for a late Form ADV filing is $15,000
```

**Correct hybrid routing:**
```
source: rag_pipeline
effective_route: HIGH_V
ANSWER: The SEC penalty for a late Form ADV filing is $25,000
```

**Monitoring alert:**
```
🚨 ALERT: SFT returned $15,000. Ground truth is $25,000 (effective 2024-02-01).
Action: reclassify this query category to HIGH_V immediately.
```

---

## Human Decision Node

During the drafting of this chapter, the AI (Bookie) proposed a three-tier router classification: HIGH_V, MEDIUM_V for annually-updating knowledge, and LOW_V. The claim was that routing MEDIUM_V queries to a cache-first RAG path would reduce unnecessary full retrievals.

**This claim is architecturally wrong.**

The router's false negative rate — HIGH_V queries misrouted to a lower tier — is the critical reliability parameter in a hybrid system. Adding MEDIUM_V creates an ambiguous boundary that increases this rate. A HIGH_V query misrouted to a MEDIUM_V cache path may return a stale answer silently, a failure mode harder to detect than a simple HIGH_V/LOW_V misroute.

The binary classifier was kept. It fails more visibly, is more auditable, and when it fails it fails toward retrieval — not toward staleness.

This decision is documented in `authors_note.md` (Page 2) and stated on camera in the video at the Human Decision Node.

---

## Figures

All four publication figures are in `figures/figures.html`. Open in any browser and print to PDF.

| Figure | Content |
|---|---|
| 13.1 | Hybrid routing architecture — both paths with latency and cost annotations |
| 13.2 | Retrieval failure in embedding space — two chunks equidistant from query vector |
| 13.3 | Volatility index table with color bands and V=1 threshold line |
| 13.4 | Cache TTL failure timeline — failure window between amendment and TTL expiry |

---

## References

- Mallen et al. (2023). *When Not to Trust Language Models.* ACL 2023.
- Kandpal et al. (2023). *Large Language Models Struggle to Learn Long-Tail Knowledge.* ICML 2023.
- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020.
- Lazaridou et al. (2021). *Mind the Gap: Assessing Temporal Generalization in Neural Language Models.* NeurIPS 2021.
- Asai et al. (2024). *Self-RAG: Learning to Retrieve, Generate, and Critique.* ICLR 2024.
- Brown, N. (2026). *80 Days to Stay.* skepticism.ai. https://www.skepticism.ai/p/80-days-to-stay-connecting-recent

---

## Author

Rohan — MS Software Engineering Systems, Northeastern University
INFO 7375 — Prompt Engineering for Generative AI
