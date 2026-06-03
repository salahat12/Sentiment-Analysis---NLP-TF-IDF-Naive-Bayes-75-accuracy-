# 💬 Sentiment Analyzer — NLP Text Classification

> **Project 4** of the DecodeLabs Industrial Training Program | Batch 2026  
> *Teaching machines to understand human emotion in text.*

---

## 📌 Overview

An NLP pipeline that classifies product reviews as **Positive**, **Negative**, or **Neutral** using TF-IDF vectorization and a Multinomial Naive Bayes classifier. Includes a live interactive analyzer with confidence scores and visual probability bars.

**Result: 75% Accuracy | Balanced across all 3 sentiment classes**

---

## 🧠 The Concept: From Numbers to Meaning

Previous projects worked with structured data (measurements, tags). Sentiment Analysis works on **unstructured text** — the messiest, richest data type.

| Project | Data Type | Engine |
|---|---|---|
| P2 — Classification | Tabular (numbers) | KNN |
| P3 — Recommendation | Tag strings | TF-IDF + Cosine |
| **P4 — Sentiment** | **Free-form text** | **TF-IDF + Naive Bayes** |

---

## ⚙️ Architecture: The NLP Pipeline

```
INPUT              PROCESS                          OUTPUT
──────────────────────────────────────────────────────────
Raw Review  →  1. Preprocess  (clean text)   →   Sentiment
reviews.csv    2. TF-IDF      (vectorize)        + Confidence
               3. Train/Split (80/20)            + Probability
               4. Naive Bayes (classify)           Breakdown
               5. Evaluate    (report)
```

---

## 🔬 Text Preprocessing Pipeline

Raw text is noisy — punctuation, capitalization, and extra spaces all add noise. The pipeline cleans it in 3 steps:

```python
def preprocess(text):
    text = text.lower()                    # "GREAT!" → "great!"
    text = re.sub(r"[^a-z\s]", "", text)  # "great!" → "great"
    text = re.sub(r"\s+", " ", text)      # clean whitespace
    return text.strip()
```

---

## 🧮 Why Multinomial Naive Bayes for Text?

Naive Bayes treats each word as an **independent probability signal**:

```
P(positive | "amazing product") ∝ P("amazing"|positive) × P("product"|positive)
```

It's the industry standard for text classification because:
- Works extremely well with TF-IDF sparse matrices
- Fast to train — no gradient descent needed
- Laplace smoothing (α=1) handles unseen words gracefully
- Interpretable — probabilities per word per class

---

## ✅ Features

- 📥 **Data Loading** — 60 balanced reviews (20 per class) from `reviews.csv`
- 🧹 **NLP Preprocessing** — lowercase, punctuation removal, whitespace cleaning
- 🔢 **TF-IDF + Bigrams** — `ngram_range=(1,2)` captures phrases like "not good"
- 🤖 **Multinomial Naive Bayes** — with Laplace smoothing for unseen words
- 📊 **Full Evaluation** — Accuracy, Confusion Matrix, Classification Report
- 🔴🟡🟢 **Live Analyzer** — Interactive loop with emoji sentiment + confidence bars

---

## 🚀 Getting Started

### Install dependencies

```bash
pip install scikit-learn pandas numpy
```

### Run the analyzer

```bash
python3 sentiment_analyzer.py
```

### Example Session

```
============================================================
  PROJECT 4 — Sentiment Analysis Using NLP
============================================================

[DATASET]  60 labeled reviews loaded
           Classes : {'positive': 20, 'negative': 20, 'neutral': 20}

[NLP PREP] Original : 'This product is absolutely amazing, I love it!'
           Cleaned  : 'this product is absolutely amazing i love it'

[TF-IDF]   Vocabulary : 358 n-gram features
[MODEL]    MultinomialNB trained  (Laplace smoothing α=1.0)

[RESULTS]  Accuracy : 75.0%

────────────────────────────────────────────────────────────
  LIVE SENTIMENT ANALYZER
────────────────────────────────────────────────────────────

Your review: This is absolutely incredible, love it!

  Sentiment  : 😊 POSITIVE  (53.7% confidence)
  Breakdown  :
      positive : ██████████            53.7%
       neutral : ████                  23.2%
      negative : ████                  23.1%
```

---

## 📁 Project Structure

```
decodelabs-ai-projects/
│
├── Project_1/   chatbot.py           Rule-Based AI
├── Project_2/   classifier.py        KNN Classification
├── Project_3/   recommender.py       TF-IDF Recommendation
│
├── Project_4/
│   ├── sentiment_analyzer.py   ← NLP sentiment pipeline
│   ├── reviews.csv             ← 60 labeled reviews dataset
│   └── README.md               ← You are here
```

---

## 🔑 Key Concepts Demonstrated

- **NLP Pipeline** — Raw text → clean → vectorize → classify
- **TF-IDF with Bigrams** — `ngram_range=(1,2)` captures "not good" as a feature
- **Multinomial Naive Bayes** — Probabilistic classifier ideal for text
- **Laplace Smoothing** — Handles words not seen during training (α=1)
- **Balanced Dataset** — Equal class distribution prevents accuracy bias
- **Stratified Split** — Maintains class ratios in train/test sets

---

## ⚠️ The Negation Problem

A key NLP challenge: "**not** good" ≠ "good". Simple unigram models miss this.  
This project uses **bigrams** (`ngram_range=(1,2)`) to partially solve it:

```
Unigram : ["not", "good"]          → treats separately
Bigram  : ["not", "good", "not good"] → captures phrase meaning
```

---

## 🏷️ Built With

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-NLP-orange?style=flat-square&logo=scikit-learn)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-red?style=flat-square)
![DecodeLabs](https://img.shields.io/badge/DecodeLabs-Batch_2026-purple?style=flat-square)

---

*"The machine does not feel. But it can learn the patterns of those who do."*  
**— DecodeLabs, Module 04**
