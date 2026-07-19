# Hate Speech Sentiment Classifier — Indonesian Twitter Data

A Multinomial Naïve Bayes text classifier for detecting religious hate speech in Indonesian-language Twitter posts. **Published in JISMAN Journal (November 2024).**

[![Publication](https://img.shields.io/badge/Publication-JISMAN%202024-blue)](https://garuda.kemdiktisaintek.go.id/documents/detail/4926421)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

## Overview

**Problem.** Religious hate speech on Indonesian social media is difficult to detect at scale. Manual moderation is slow, and off-the-shelf English NLP models miss Indonesian morphology and slang.

**Approach.** Trained a Multinomial Naïve Bayes classifier on a labeled dataset of Indonesian tweets. Built a full NLP preprocessing pipeline tailored for Bahasa Indonesia (tokenization, stopword removal with Sastrawi, stemming), then encoded features with Bag-of-Words and tuned the Laplace smoothing parameter (alpha) via cross-validation.

**Results.**

| Evaluation           | Accuracy |
| -------------------- | -------- |
| Holdout test set     | **85%**  |
| 10-fold cross-val    | **79%**  |

Published in JISMAN Journal, November 2024 — [read the paper](https://garuda.kemdiktisaintek.go.id/documents/detail/4926421).

## Tech Stack

- **Python** — core language
- **scikit-learn** — Multinomial Naïve Bayes, CountVectorizer, cross-validation
- **Sastrawi** — Indonesian stemmer and stopword remover
- **NLTK** — tokenization
- **Flask** — lightweight web app for live predictions
- **pandas / openpyxl** — dataset handling

## Project Structure

```
KlasifikasiSentimenTwitter/
├── app.py                 # Flask web app for live prediction
├── preprocessing.py       # Text cleaning pipeline (tokenize, stopwords, stem)
├── bow.py                 # Bag-of-Words feature extraction
├── mnb.py                 # Multinomial Naive Bayes model
├── train_model.py         # Model training entrypoint
├── find_best_alpha.py     # Hyperparameter tuning for Laplace smoothing
├── kfold_cv.py            # 10-fold cross-validation runner
├── dataset.xlsx           # Labeled tweet dataset
├── pickle/                # Serialized trained model + vectorizer
├── csv/                   # Intermediate CSVs from preprocessing
└── requirements.txt
```

## Getting Started

**1. Clone the repo**

```bash
git clone https://github.com/aljuhaeda/KlasifikasiSentimenTwitter.git
cd KlasifikasiSentimenTwitter
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Train the model**

```bash
python train_model.py
```

This runs the preprocessing pipeline, fits the Multinomial Naïve Bayes model, and saves the model + vectorizer to `pickle/`.

**4. Run the web app**

```bash
python app.py
```

Then open `http://localhost:5000` and paste a tweet to classify.

## Reproducing the Paper Results

- 10-fold cross-validation: `python kfold_cv.py`
- Alpha (Laplace smoothing) tuning: `python find_best_alpha.py`

## Publication

> Al Juhaeda, Z. I. (2024). *Sentiment Analysis of Hate Speech in Twitter Comments Using Naïve Bayes Classifier.* JISMAN Journal.
> [View on Garuda / Kemdiktisaintek](https://garuda.kemdiktisaintek.go.id/documents/detail/4926421)

## License

MIT. See [LICENSE](LICENSE).

## Author

**Zul Iflah Al Juhaeda** — [LinkedIn](https://linkedin.com/in/aljuhaeda) · [GitHub](https://github.com/aljuhaeda)
