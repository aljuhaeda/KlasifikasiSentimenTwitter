# KlasifikasiSentimenTwitter — Hate Speech Detection Against Islam on Twitter

Multinomial Naive Bayes classifier detecting hate speech directed at Islam in Indonesian-language tweets, with systematic alpha tuning and k-fold cross-validation. Basis for a peer-reviewed publication.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![App screenshot](docs/app-screenshot.png)

## Overview

**Problem.** Automatically flagging hate speech against religious groups on social media is a narrow, high-stakes text classification task: the signal is sparse (most tweets aren't hate speech), the language is informal Indonesian Twitter text, and a false negative (missing real hate speech) is more costly than a false positive.

**Approach.**

1. Collected and labeled ~1,200 Indonesian-language tweets as **Negatif** (no hate speech) or **Positif** (hate speech present).
2. Preprocessed text: URL/mention/hashtag stripping, case folding, tokenization, Indonesian stopword removal, stemming.
3. The notebook also derives lexicon-based labels from `lexicon/positive.tsv` / `lexicon/negative.tsv` (written to `csv/*_with_classes.csv`), but these are **not used in training**: the model is trained on the manual labels with Bag-of-Words features only.
4. Vectorized with `CountVectorizer` and trained **Multinomial Naive Bayes**, systematically tuning the smoothing parameter `alpha` (0.01 → 3.0) across three train/test splits (70/30, 80/20, 90/10).
5. Validated the strongest configurations with **10-fold cross-validation**.

Full walkthrough is in [`skripsi.ipynb`](skripsi.ipynb).

## Results

Headline accuracy tops out at **84%** (`alpha=3.0`, 70/30 split), but accuracy is the wrong number to optimize here — see Key Finding below. The demo app ships with **`alpha=2.0`, 80/20 split**:

| Config | Accuracy | Positif (hate speech) Recall | Positif F1 |
|---|---|---|---|
| alpha=3.0, 70/30 (highest accuracy) | 0.84 | 0.14 | 0.21 |
| alpha=2.0, 70/30 | 0.84 | 0.20 | 0.29 |
| **alpha=2.0, 80/20 (deployed)** | 0.83 | **0.18** | **0.25** |
| alpha=1.0, 80/20 | 0.82 | 0.26 | 0.32 |

The 70/30 and 80/20 rows are scored on different test sets, so they are not directly comparable. On the same 70/30 test set, alpha=2.0 has the same accuracy as alpha=3.0 (0.8352 for both) with higher Positif recall (0.20 vs 0.14). These are original-pipeline figures; see the **Reproducibility note** below for leak-free numbers.

Full sweep across every alpha × split combination tested is in `csv/accuracy_*.csv` and `csv/classification_report_*.csv`.

## Key Finding / Limitation

**Class imbalance is the real bottleneck, not the algorithm — and optimizing for accuracy alone makes the actual problem worse.** The alpha that maximizes headline accuracy (3.0) has the *worst* recall on the `Positif` (hate speech) class of any tuned configuration — it does well by predicting `Negatif` almost every time, since that's the majority class. A hate-speech filter that catches 14% of real hate speech is not a useful filter, regardless of its accuracy score. This tension — and the fact that the deployed config still misses roughly 4 in 5 real hate-speech tweets — is the central limitation, documented in the published paper, not glossed over.

## Reproducibility note

The figures in the published paper (JISMEDIA, July 2024) and in the Results table above come from the **original pipeline** and are kept as published. A 2026-11-03 audit of this repo found three issues in that pipeline that make its test-set numbers optimistic:

1. **Vectorizer fit includes test rows.** The `CountVectorizer` (1,000-word vocabulary) is fit once on the union of all three ratios' training sets. The three splits are different partitions, so 238 of the 358 test rows at 70:30 and 119 of the 239 at 80:20 are training rows of another ratio and were in the vectorizer's fit corpus. Only the vocabulary leaked, not labels. Counting rows whose text matches a fit-corpus row, the figures are 241/358, 122/239 and 3/120 (90:10).
2. **Duplicates across the split.** After preprocessing, 7/358, 4/239 and 3/120 test rows have the same text as a row in their training set.
3. **Alpha chosen on the test set.** `find_best_alpha` selects alpha by test accuracy; there is no separate validation step.

`eval_leakfree.py` re-runs the evaluation without these issues: same stratified splits (`random_state=42`), vectorizer fit on each training portion only, duplicate test rows removed, alpha chosen by 10-fold CV on the training portion only (grid 0.01, 0.1, 0.4, 0.5, 0.6, 0.8, 1.0, 2.0, 3.0), test set used once. The 10-fold CV refits the vectorizer inside each fold. Output: [`results_leakfree.txt`](results_leakfree.txt) (Python 3.14.7, scikit-learn 1.9.0).

| Setting | As published / original pipeline | Leak-free re-run |
|---|---|---|
| 90:10, alpha=1.0 (paper's Model F), accuracy | 85% (paper); 0.78 in this repo's committed `csv/` | 0.7949 (117 test rows) |
| 90:10, alpha=1.0, Positif precision / recall / F1 | 60 / 16 / 25% (paper, Table 9) | 0.22 / 0.11 / 0.14 |
| 10-fold CV mean accuracy | 79.09% (paper) | 0.8085 (90:10 training portion, alpha=1.0) |
| 80:20, alpha=2.0 (deployed config), accuracy | 0.83 | 0.8170 (235 test rows) |
| 80:20, alpha=2.0, Positif precision / recall / F1 | 0.44 / 0.18 / 0.25 | 0.33 / 0.13 / 0.19 |
| Alpha picked by CV accuracy, per ratio | — | 3.0 for all three ratios; test accuracy 0.8319 / 0.8298 / 0.8205 (70:30 / 80:20 / 90:10), Positif recall 0.09 / 0.11 / 0.11 |

The paper's 85% (90:10, alpha=1.0) and 79.09% CV figures are not reproduced by this repo's current code. The committed results for that setting give 0.78, and a fresh re-run of the original code gives 0.79. The paper's results appear to come from a different run or configuration than the one committed here.

**Label definition.** The paper (Section 2.1, Table 1) defines `Positif` as non-hate speech and `Negatif` as hate speech against Islam. This README describes `Positif` as hate speech. The two descriptions are opposite, and the owner has not yet confirmed which one holds for `dataset.xlsx`. All numbers here are reported per label name.

Full detail: [`ERRATA.md`](ERRATA.md) and [`PROGRESS.md`](PROGRESS.md).

## Publication

Al Juhaeda, Z. I., Faisal, M., & Suhartono. (2024). *Sentiment Classification of Hate Speech Against Islam on Twitter Platform Using Multinomial Naïve Bayes.* Journal of Informatics and Science Media (JISMEDIA), 1(1), 34–38. [View on Garuda](https://garuda.kemdiktisaintek.go.id/documents/detail/4926421)

## Tech Stack

- **Python**, **Jupyter Notebook**
- **scikit-learn** — CountVectorizer, MultinomialNB, KFold cross-validation
- **NLTK** — Indonesian stopword list; stemming uses NLTK's English `PorterStemmer`, applied as-is to Indonesian text (no Indonesian stemmer). The training pipeline (`preprocessing.py`) tokenizes with `str.split()`; the demo app (`app.py`) tokenizes with NLTK `word_tokenize`.
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

**Note on `train_model.py`:** this is a leftover standalone script with hardcoded local file paths from the original thesis machine (`C:\SkripsiFix\...`) — it's not part of the reproducible flow above. `skripsi.ipynb` still has the same hardcoded `C:\SkripsiFix\...` paths in several cells (a path-swap to relative paths, matching what the deployed `app.py` already does via `os.path.join(base_dir, ...)`, hasn't been done here) — update those paths for your own machine before rerunning the notebook.

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

- [x] Preprocessing (lexicon labels computed in the notebook, not used in training)
- [x] Alpha tuning across 3 train/test splits
- [x] 10-fold cross-validation
- [x] Streamlit demo app
- [x] Published (JISMEDIA Vol. 1 Iss. 1, July 2024)
- [x] Deployed (Streamlit Community Cloud)

## License

MIT — see [LICENSE](LICENSE).

## Author

**Zul Iflah Al Juhaeda** — [LinkedIn](https://linkedin.com/in/aljuhaeda) · [GitHub](https://github.com/aljuhaeda)
