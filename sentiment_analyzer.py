# ============================================================
#  Project 4 – Sentiment Analysis Using NLP
#  DecodeLabs | Industrial Training Kit | Batch 2026
#  Method  : NLP Text Pipeline + TF-IDF + Multinomial Naive Bayes
#  Dataset : reviews.csv  (36 labeled reviews, 3 sentiment classes)
# ============================================================

import re
import pandas as pd
from sklearn.feature_extraction.text  import TfidfVectorizer
from sklearn.naive_bayes              import MultinomialNB
from sklearn.model_selection          import train_test_split
from sklearn.metrics                  import (classification_report,
                                               confusion_matrix,
                                               accuracy_score)

# ── PHASE 1 : INPUT — Load & Preprocess ─────────────────────
print("=" * 60)
print("  PROJECT 4 — Sentiment Analysis Using NLP")
print("=" * 60)

df = pd.read_csv("reviews.csv")
print(f"\n[DATASET]  {len(df)} labeled reviews loaded")
print(f"           Classes : {df['sentiment'].value_counts().to_dict()}")

# ── TEXT PREPROCESSING PIPELINE ─────────────────────────────
def preprocess(text: str) -> str:
    """
    NLP cleaning pipeline:
    1. Lowercase          → normalize case
    2. Remove punctuation → reduce noise
    3. Strip extra spaces → clean whitespace
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_review"] = df["review"].apply(preprocess)
print(f"\n[NLP PREP] Text preprocessing applied:")
print(f"           Original : '{df['review'].iloc[0]}'")
print(f"           Cleaned  : '{df['clean_review'].iloc[0]}'")

# ── PHASE 2 : PROCESS — Vectorize → Split → Train ───────────

# Step 1 — TF-IDF Vectorization (same engine as Project 3)
#           stop_words removes common words: "the", "is", "a" etc.
vectorizer   = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
X            = vectorizer.fit_transform(df["clean_review"])
y            = df["sentiment"]
print(f"\n[TF-IDF]   Vocabulary : {len(vectorizer.vocabulary_)} n-gram features")
print(f"           Matrix     : {X.shape}  (reviews × features)")

# Step 2 — Train / Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[SPLIT]    Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# Step 3 — Multinomial Naive Bayes
#           Best suited for TF-IDF text features — assumes
#           word probabilities given each sentiment class
model = MultinomialNB(alpha=1.0)   # alpha = Laplace smoothing
model.fit(X_train, y_train)
print(f"\n[MODEL]    MultinomialNB trained  (Laplace smoothing α=1.0)")

# ── PHASE 3 : OUTPUT — Evaluate & Predict ───────────────────
predictions = model.predict(X_test)
accuracy    = accuracy_score(y_test, predictions)

print(f"\n[RESULTS]")
print(f"  Accuracy   : {accuracy * 100:.1f}%")

print(f"\n[CONFUSION MATRIX]")
labels = ["negative", "neutral", "positive"]
cm     = confusion_matrix(y_test, predictions, labels=labels)
header = "          " + "  ".join(f"{l:>10}" for l in labels)
print(f"  {header}")
for i, row in enumerate(cm):
    print(f"  {labels[i]:>10}  " + "  ".join(f"{v:>10}" for v in row))

print(f"\n[CLASSIFICATION REPORT]")
print(classification_report(y_test, predictions, target_names=labels))

# ── LIVE SENTIMENT PREDICTOR ─────────────────────────────────
print("─" * 60)
print("  LIVE SENTIMENT ANALYZER — Type a review to analyze")
print("  (type 'quit' to exit)")
print("─" * 60)

SENTIMENT_EMOJI = {
    "positive" : "😊 POSITIVE",
    "neutral"  : "😐 NEUTRAL",
    "negative" : "😠 NEGATIVE"
}

while True:
    user_review = input("\nYour review: ").strip()
    if user_review.lower() in ("quit", "exit", "q"):
        print("Goodbye! 🚀")
        break
    if not user_review:
        continue

    # Preprocess → Vectorize → Predict
    cleaned    = preprocess(user_review)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    proba      = model.predict_proba(vectorized)[0]
    classes    = model.classes_

    sentiment_label = SENTIMENT_EMOJI[prediction]
    confidence      = max(proba) * 100

    print(f"\n  Sentiment  : {sentiment_label}  ({confidence:.1f}% confidence)")
    print(f"  Breakdown  :", end="")
    for cls, p in sorted(zip(classes, proba), key=lambda x: -x[1]):
        bar = "█" * int(p * 20)
        print(f"\n    {cls:>10} : {bar:<20}  {p*100:.1f}%")
