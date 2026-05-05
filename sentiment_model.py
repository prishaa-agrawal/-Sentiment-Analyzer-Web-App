"""
Sentiment Analysis Model
Uses TF-IDF + Logistic Regression trained on a small labeled dataset.
Falls back to VADER lexicon for demo purposes if sklearn is unavailable.
"""

import re
import os
import pickle
import numpy as np

# Try loading sklearn; graceful fallback to lexicon-based approach
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Tiny hand-crafted training set (enough to demo — real project uses a CSV)
# ---------------------------------------------------------------------------
SAMPLE_DATA = [
    ("I love this product! It's absolutely amazing.", "positive"),
    ("This is the best thing I've ever bought.", "positive"),
    ("Great quality, fast shipping, very happy.", "positive"),
    ("Excellent service and wonderful experience.", "positive"),
    ("Fantastic! Totally exceeded my expectations.", "positive"),
    ("I'm so happy with this purchase.", "positive"),
    ("Really good, would buy again.", "positive"),
    ("Outstanding performance and great value.", "positive"),
    ("Terrible product, complete waste of money.", "negative"),
    ("I hate this, it broke after one day.", "negative"),
    ("Worst purchase ever, do not buy.", "negative"),
    ("Awful quality, very disappointed.", "negative"),
    ("Does not work at all, very frustrating.", "negative"),
    ("Poor customer service and bad product.", "negative"),
    ("Really bad experience, would not recommend.", "negative"),
    ("Total garbage, nothing works as described.", "negative"),
    ("It's okay, nothing special.", "neutral"),
    ("Average product, meets basic expectations.", "neutral"),
    ("Not great, not terrible, just mediocre.", "neutral"),
    ("Does the job, but nothing impressive.", "neutral"),
    ("Decent enough for the price.", "neutral"),
    ("Works as expected, no complaints.", "neutral"),
]


def clean_text(text: str) -> str:
    """Basic text cleaning pipeline."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"[^a-z0-9\s!?.,']", " ", text)       # keep punctuation signals
    text = re.sub(r"\s+", " ", text).strip()
    return text


class SentimentAnalyzer:
    """
    Wraps a TF-IDF + Logistic Regression pipeline.
    Trains on sample data at init; swap in your own CSV for production.
    """

    LABEL_EMOJI = {
        "positive": "😊",
        "negative": "😞",
        "neutral":  "😐",
    }

    LABEL_COLOR = {
        "positive": "#22c55e",
        "negative": "#ef4444",
        "neutral":  "#f59e0b",
    }

    def __init__(self, model_path: str = None):
        self.model = None
        self._label_map = {0: "negative", 1: "neutral", 2: "positive"}

        if model_path and os.path.exists(model_path):
            self._load(model_path)
        else:
            self._train()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _train(self):
        if not SKLEARN_AVAILABLE:
            print("[SentimentAnalyzer] sklearn not found — using lexicon fallback.")
            return

        texts, labels = zip(*SAMPLE_DATA)
        texts = [clean_text(t) for t in texts]

        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=1.0,
                multi_class="multinomial",
                solver="lbfgs",
            )),
        ])
        self.model.fit(texts, labels)

    def _load(self, path: str):
        with open(path, "rb") as f:
            self.model = pickle.load(f)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, text: str) -> dict:
        cleaned = clean_text(text)

        if self.model and SKLEARN_AVAILABLE:
            label = self.model.predict([cleaned])[0]
            proba = self.model.predict_proba([cleaned])[0]
            classes = self.model.classes_
            confidence = float(np.max(proba))
            scores = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
        else:
            label, confidence, scores = self._lexicon_predict(cleaned)

        return {
            "label":      label,
            "confidence": round(confidence * 100, 1),
            "scores":     scores,
            "emoji":      self.LABEL_EMOJI[label],
            "color":      self.LABEL_COLOR[label],
            "text":       text,
        }

    # ------------------------------------------------------------------
    # Simple lexicon fallback (no sklearn required)
    # ------------------------------------------------------------------
    _POS_WORDS = {"good","great","love","excellent","amazing","fantastic",
                  "wonderful","best","happy","awesome","superb","brilliant",
                  "outstanding","perfect","enjoy","positive","recommend"}
    _NEG_WORDS = {"bad","terrible","hate","awful","worst","horrible","poor",
                  "disappointing","useless","broken","waste","negative","ugly",
                  "disgusting","boring","dreadful","pathetic"}

    def _lexicon_predict(self, text: str):
        words = set(text.split())
        pos = len(words & self._POS_WORDS)
        neg = len(words & self._NEG_WORDS)
        total = pos + neg + 1e-9
        if pos > neg:
            label = "positive"
            conf  = min(0.55 + pos / total * 0.4, 0.99)
        elif neg > pos:
            label = "negative"
            conf  = min(0.55 + neg / total * 0.4, 0.99)
        else:
            label = "neutral"
            conf  = 0.55
        scores = {
            "positive": round(pos / total, 4),
            "negative": round(neg / total, 4),
            "neutral":  round(1 - pos / total - neg / total, 4),
        }
        return label, conf, scores
