#  SentiScope — NLP Sentiment Analyzer

A beginner-friendly **end-to-end NLP project** that classifies text as **Positive**, **Negative**, or **Neutral** using a TF-IDF + Logistic Regression pipeline, served via a sleek Flask web app.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.3%2B-lightgrey)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

##  Features

-  **ML Pipeline** — TF-IDF vectorizer + Logistic Regression with bigrams
-  **Web App** — Clean Flask UI with animated confidence score bars
-  **Jupyter Notebook** — Full EDA, training, and evaluation walkthrough
-  **REST API** — `/predict` and `/batch` endpoints (JSON)
-  **Lexicon Fallback** — Works even without scikit-learn installed

---

##  Project Structure
```
sentiment-analyzer/
│
├── app.py                        # Flask web application
├── requirements.txt              # Python dependencies
│
├── model/
│   ├── __init__.py
│   └── sentiment_model.py        # ML pipeline (TF-IDF + LR)
│
├── templates/
│   └── index.html                # Frontend UI
│
└── notebooks/
    └── training_walkthrough.ipynb  # EDA + model training
```

---

##  Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/sentiment-analyzer.git
cd sentiment-analyzer
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python app.py
```

Open your browser at `http://localhost:5000` 

---

##  API Usage

### Single Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely amazing!"}'
```

**Response:**
```json
{
  "label": "positive",
  "confidence": 94.2,
  "scores": { "positive": 0.942, "neutral": 0.041, "negative": 0.017 },
  "emoji": "😊",
  "color": "#22c55e",
  "text": "This product is absolutely amazing!"
}
```

### Batch Prediction
```bash
curl -X POST http://localhost:5000/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great!", "Terrible.", "It is okay."]}'
```

---

##  How It Works

| Step | Description |
|------|-------------|
| **Preprocessing** | Lowercasing, URL removal, special char stripping |
| **Vectorization** | TF-IDF with unigrams + bigrams, sublinear TF scaling |
| **Classification** | Logistic Regression (multinomial, L2 regularization) |
| **Output** | Label + per-class probability scores |

---

##  Improving the Model

Want better accuracy? Swap the sample data with a real dataset:

- [IMDb Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/)
- [Amazon Product Reviews](https://jmcauley.ucsd.edu/data/amazon/)
- [Twitter Sentiment140](http://help.sentiment140.com/for-students)

Load it in `sentiment_model.py` by replacing `SAMPLE_DATA` with a CSV read.

---

## Tech Stack

- **Backend:** Python, Flask
- **ML:** scikit-learn (TF-IDF, Logistic Regression)
- **Frontend:** Vanilla HTML/CSS/JS
- **Notebook:** Jupyter




