# KlasifikasiSentimenTwitter — Hate Speech Detection Against Islam on Twitter

Multinomial Naive Bayes classifier detecting hate speech directed at Islam in Indonesian-language tweets, with a lexicon-based augmentation step, systematic alpha tuning, and k-fold cross-validation. Basis for a peer-reviewed publication.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

**Problem.** Automatically flagging hate speech against religious groups on social media is a narrow, high-stakes text classification task: the signal is sparse (most tweets aren't hate speech), the language is informal Indonesian Twitter text, and a false negative (missing real hate speech) is more costly than a false positive.

**Approach.**

1. Collected and labeled ~1,200 Indonesian-language tweets as **Negatif** (no hate speech) or **Positif** (hate speech present).
2. Preprocessed text: URL/mention/hashtag stripping, case folding, tokenization, Indonesian stopword removal, stemming.
3. Augmented the model with a **positive/negative sentiment lexicon** (`lexicon/positive.tsv`, `lexicon/negative.tsv`) alongside the Bag-of-Words features.
4. Vectorized with `CountVectorizer` and trained **Multinomial Naive Bayes**, systematically tuning the smoothing parameter `alpha` (0.01 → 3.0) across three train/test splits (70/30, 80/20, 90/10).
5. Validated the strongest configurations with **10-fold cross-validation**.

Full walkthrough is in [`skripsi.ipynb`](skripsi.ipynb).

## Results

Best accuracy — **84%** — reached at `alpha=2.0`–`3.0` on the 70/30 split (see `csv/accuracy_*.csv` for the full sweep across every alpha × split combination tested).

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Negatif (no hate speech) | 0.87 | 0.93 | 0.90 |
| Positif (hate speech) | 0.42 | 0.26 | 0.32 |

*(alpha=1.0, 80/20 split — one of the tuned configurations, not the best-accuracy one; shown because it's the config the demo app ships with.)*

## Key Finding / Limitation

**Class imbalance is the real bottleneck, not the algorithm.** Overall accuracy (82–84%) looks strong, but it's propped up by the majority `Negatif` class — recall on `Positif` (the class that actually matters for a hate-speech filter) tops out around 0.26–0.32 across most configurations. In other words: the model catches the *absence* of hate speech reliably but misses a majority of the *actual* hate speech in the test set. This is documented and discussed as the central limitation in the published paper (see below), not glossed over.

## Publication

Al Juhaeda, Z. I. (2024). *Sentiment Analysis of Hate Speech in Twitter Comments Using Naïve Bayes Classifier.* JISMAN Journal. [View on Garuda](https://garuda.kemdiktisaintek.go.id/documents/detail/4926421)

## Tech Stack

- **Python**, **Jupyter Notebook**
- **scikit-learn** — CountVectorizer, MultinomialNB, KFold cross-validation
- **NLTK** — tokenization, Indonesian stopword removal, stemming
- **Streamlit** — interactive demo app
- **pandas / NumPy** — data handling

## Demo

**Live**: [klasifikasisentimentwitter.streamlit.app](https://klasifikasisentimentwitter.streamlit.app/) — paste a tweet, get a Negatif/Positif prediction with class probabilities, plus an expandable breakdown of every preprocessing step (cleaning → tokenizing → stopword removal → stemming → Bag-of-Words) and the top contributing words per class.

Run it locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repo Structure

```
KlasifikasiSentimenTwitter/
├── skripsi.ipynb        # Full walkthrough: EDA, preprocessing, alpha tuning, CV, evaluation
├── app.py                # Streamlit demo app
├── preprocessing.py       # Text cleaning / tokenizing / stopword removal / stemming
├── bow.py                 # CountVectorizer training + Bag-of-Words transforms
├── mnb.py                 # MultinomialNB train / evaluate / save helpers
├── kfold_cv.py             # K-fold cross-validation
├── find_best_alpha.py      # Alpha sweep result comparison
├── train_model.py          # Legacy standalone retrain script (has machine-specific paths — see note below)
├── lexicon/                # positive.tsv / negative.tsv sentiment lexicons
├── dataset.xlsx            # Labeled tweet dataset (~1,200 rows: full_text, kelas)
├── pickle/                 # Trained models + vectorizers per alpha/split combination
├── csv/                    # Accuracy + classification-report results per alpha/split combination
└── requirements.txt
```

**Note on `train_model.py`:** this is a leftover standalone script with hardcoded local file paths from the original thesis machine (`C:\SkripsiFix\...`) — it's not part of the reproducible flow above. Reproduce results via `skripsi.ipynb`, which uses relative paths.

## Reproduction

```bash
git clone https://github.com/aljuhaeda/KlasifikasiSentimenTwitter.git
cd KlasifikasiSentimenTwitter
pip install -r requirements.txt
```

Open `skripsi.ipynb` and run all cells — it downloads required NLTK resources (`punkt`, `stopwords`) automatically.

## Limitations

- **Small dataset** (~1,200 tweets) for a 2-class problem with real class imbalance — limits how well the minority (`Positif`) class can be learned.
- **Single platform, single language variant** — Indonesian-language Twitter text only; informal/slang coverage depends entirely on what's in the training set and lexicon.
- **Bag-of-Words + Naive Bayes** has no notion of word order or context — sarcasm, negation, and indirect hate speech are hard cases this architecture structurally can't catch.
- See **Key Finding** above — low recall on the `Positif` class is the main limitation, not a footnote.

## Status

- [x] Preprocessing + lexicon augmentation
- [x] Alpha tuning across 3 train/test splits
- [x] 10-fold cross-validation
- [x] Streamlit demo app
- [x] Published (JISMAN Journal, Nov 2024)
- [x] Deployed (Streamlit Community Cloud)

## License

MIT — see [LICENSE](LICENSE).

## Author

**Zul Iflah Al Juhaeda** — [LinkedIn](https://linkedin.com/in/aljuhaeda) · [GitHub](https://github.com/aljuhaeda)
