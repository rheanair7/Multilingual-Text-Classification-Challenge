# SemEval 2026 Task 9 – Subtask 1  
Group Members: Rhea Nair & Kruti Shah
### Detecting Multilingual, Multicultural and Multievent Online Polarization

This repository contains our work for **SemEval 2026 Task 9 (POLAR)**.  
We are currently implementing **Subtask 1: Binary Polarization Detection**, and we will also be extending our system to support:

- **Subtask 2 – Fine-Grained Polarization Classification**  
- **Subtask 3 – Multilingual Event-Based Polarization Detection**

We focus on building **per-language transformer models** using:
- **XLM-RoBERTa-Base** (completed for 6 languages)
- **mDeBERTa-v3-Base** (optimized configuration, training in progress)

Our work emphasizes **multilingual generalization**, **handling low-resource languages**, and **evaluating macro-F1 across languages**.

---

## Task Overview

Subtask 1 involves **binary classification**:

| Label | Meaning |
|-------|---------|
| **1** | The text expresses polarized content (hostility, groupism, extreme stance, stereotyping, etc.) |
| **0** | Non-polarized |

We use only what is expressed **directly in the text** — not:
- the reader’s reaction,  
- someone else’s quoted opinion,  
- or the assumed true belief of the author.

---

## Languages We Worked On

Although the organizers released 13+ languages in total,  
**our project currently focuses on the following 6 languages:**

- English (**eng**)
- Hindi (**hin**)
- Nepali (**nep**) — only supported by XLM-R
- Urdu (**urd**)
- Chinese (**zho**)
- Italian (**ita**)
- Spanish (**spa**) — only supported by mDeBERTa  
- Arabic (**arb**) — only supported by mDeBERTa  
---

## Dataset Handling

From the official **dev_phase** dataset:
- We loaded each language's *train* and *dev* CSVs.
- Ensured consistent label mapping across languages.
- Split data as required for validation.
- Saved dev-set predictions for all languages.

All datasets are stored inside this repo under:

dev_phase/subtask1/train
dev_phase/subtask1/dev

---

## What We Have Completed So Far

### 1. **End-to-End Training Pipeline Built**
- Tokenization using HuggingFace   
- Trainer configuration (per-language)  
- Logging per-epoch loss and macro-F1  
- Generating predictions (`*_dev_predicted.csv`)

---

### 2. **XLM-RoBERTa-Base Models Fully Trained (Completed)**

We trained XLM-R on **each language independently for 3 epochs**.

**Final macro-F1 Scores:**

| Language | XLM-R Base F1 |
|----------|---------------|
| eng | 0.7373 |
| hin | 0.6994 |
| nep | 0.8657 |
| urd | 0.4099 |
| zho | 0.8294 |
| ita | 0.4831 |

These results establish our **baseline** and help identify which languages are challenging (e.g., Urdu).

---

### 3. **mDeBERTa-v3-Base Experiments  
We trained multiple configurations with different learning rates, epochs, warmup ratios, weight decay, cosine scheduling, and fp16.

## mDeBERTa-v3-Base – Results by Learning Rate & Epoch Setting
| Run | Learning Rate | Epochs | eng | hin | spa | urd | zho | arb |
|-----|---------------|--------|------|------|------|------|------|------|
| **Run 1** | 2e-5 | 5 | 0.788822 | 0.756033 | 0.726920 | 0.704494 | 0.858504 | 0.768262 |
| **Run 2** | 1e-5 | 5 | 0.791830 | 0.724999 | 0.706230 | 0.721408 | 0.835746 | 0.768296 |
| **Run 3** | 1e-5 | 10 | 0.783977 | 0.737468 | 0.740693 | 0.707146 | 0.867817 | 0.776681 |
| **Run 4** | 1e-5 (optimized schedule) | 5 | 0.776085 | 0.739867 | 0.711873 | 0.710498 | 0.838041 | 0.765612 |
| **Run 5** | 1e-5 (optimized + fp16) | 5 | **0.783193** | **0.765854** | **0.739415** | **0.745109** | **0.862113** | **0.782544** |

mDeBERTa dramatically improves performance on **low-resource languages**, especially **Urdu**.
---

# Evaluation Metric

We use **Macro F1-score**, required by SemEval.  
This ensures balanced evaluation even when classes are imbalanced.

---

# Next Steps
- Perform error analysis per language 
- Perform Subtask 2 & 3
