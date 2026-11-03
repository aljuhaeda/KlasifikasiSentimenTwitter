# KlasifikasiSentimenTwitter — Hate Speech Detection Against Islam on Twitter

Multinomial Naive Bayes classifier detecting hate speech directed at Islam in Indonesian-language tweets, with systematic alpha tuning and k-fold cross-validation. Basis for a peer-reviewed publication.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![App screenshot](docs/app-screenshot.png)

## Overview

**Problem.** Automatically flagging hate speech against religious groups on social media is a narrow, high-stakes text classification task: the classes are imbalanced (in this dataset hate speech is the majority, 83.5% of rows, so the minority class is non-hate tweets about Islam), the language is informal Indonesian Twitter text, and a false negative (missing real hate speech) is more costly than a false positive.

**Approach.**

1. Collected and labeled ~1,200 Indonesian-language tweets as **Negatif** (hate speech against Islam; 1,002 rows, 83.5%) or **Positif** (non-hate speech; 198 rows, 16.5%), following the paper's label definition (Section 2.1, Table 1).
2. Preprocessed text: URL/mention/hashtag stripping, case folding, tokenization, Indonesian stopword removal, stemming.
3. The notebook also derives lexicon-based labels from `lexicon/positive.tsv` / `lexicon/negative.tsv` (written to `csv/*_with_classes.csv`), but these are **not used in training**: the model is trained on the manual labels with Bag-of-Words features only.
4. Vectorized with `CountVectorizer` and trained **Multinomial Naive Bayes**, systematically tuning the smoothing parameter `alpha` (0.01 → 3.0) across three train/test splits (70/30, 80/20, 90/10).
5. Validated the strongest configurations with **10-fold cross-validation**.

Full walkthrough is in [`skripsi.ipynb`](skripsi.ipynb).

## Results

Headline accuracy tops out at **84%** (`alpha=3.0`, 70/30 split), but accuracy is the wrong number to optimize here — see Key Finding below. The demo app ships with **`alpha=2.0`, 80/20 split**:

| Config | Accuracy | Positif (non-hate) Recall | Positif (non-hate) F1 | Negatif (hate) Precision / Recall |
|---|---|---|---|---|
| alpha=3.0, 70/30 (highest accuracy, tied with alpha=2.0) | 0.84 | 0.14 | 0.21 | 0.85 / 0.97 |
| alpha=2.0, 70/30 | 0.84 | 0.20 | 0.29 | 0.86 / 0.96 |
| **alpha=2.0, 80/20 (deployed)** | 0.83 | **0.18** | **0.25** | 0.86 / 0.96 |
| alpha=1.0, 80/20 | 0.82 | 0.26 | 0.32 | 0.87 / 0.93 |

The 70/30 and 80/20 rows are scored on different test sets, so they are not directly comparable. On the same 70/30 test set, alpha=2.0 has the same accuracy as alpha=3.0 (0.8352 for both) with higher Positif recall (0.20 vs 0.14). These are original-pipeline figures; see the **Reproducibility note** below for leak-free numbers.

Full sweep across every alpha × split combination tested is in `csv/accuracy_*.csv` and `csv/classification_report_*.csv`.

## Key Finding / Limitation

**Class imbalance is the real bottleneck, not the algorithm — and optimizing for accuracy alone makes the actual problem worse.** `Negatif` (hate speech) is the majority class, and the model leans heavily towards it. Hate-speech recall is high: 0.96 for the deployed alpha=2.0/80:20 config with precision 0.86 (original pipeline), and 0.9492 with precision 0.8500 when that config is re-fit leak-free. The minority `Positif` (non-hate) class is mostly misclassified as hate: the deployed config recognises only 0.18 of non-hate tweets (0.1316 leak-free), and the alpha that maximizes headline accuracy (3.0) recognises 0.14 at 70/30 (0.09–0.11 leak-free). So the practical limitation is false alarms: roughly 4 in 5 non-hate tweets about Islam get flagged as hate speech. Accuracy hides this, because always predicting `Negatif` already scores 0.833–0.838 on these test sets, and no configuration in this repo's committed runs beats that baseline (the paper's unreproduced 85% is the exception): the original 70/30 alpha=2.0/3.0 equal it (0.8352), as does 80/20 alpha=3.0 (0.8368), the deployed config is below it (0.8285 vs 0.8368), and every leak-free accuracy is at or below it (see Reproducibility note).

## Reproducibility note

The figures in the published paper (JISMEDIA, July 2024) and in the Results table above come from the **original pipeline** and are kept as published. A 2026-11-03 audit of this repo found three issues in that pipeline that make its test-set numbers optimistic:

1. **Vectorizer fit includes test rows.** The `CountVectorizer` (1,000-word vocabulary) is fit once on the union of all three ratios' training sets. The three splits are different partitions, so 238 of the 358 test rows at 70:30 and 119 of the 239 at 80:20 are training rows of another ratio and were in the vectorizer's fit corpus. Only the vocabulary leaked, not labels. Counting rows whose text matches a fit-corpus row, the figures are 241/358, 122/239 and 3/120 (90:10).
2. **Duplicates across the split.** After preprocessing, 7/358, 4/239 and 3/120 test rows have the same text as a row in their training set.
3. **Alpha chosen on the test set.** `find_best_alpha` selects alpha by test accuracy; there is no separate validation step.

`eval_leakfree.py` re-runs the evaluation without these issues: same stratified splits (`random_state=42`), vectorizer fit on each training portion only, duplicate test rows removed, alpha chosen by 10-fold CV on the training portion only (grid 0.01, 0.1, 0.4, 0.5, 0.6, 0.8, 1.0, 2.0, 3.0), test set used once. The 10-fold CV refits the vectorizer inside each fold. Output: [`results_leakfree.txt`](results_leakfree.txt) (Python 3.14.7, scikit-learn 1.9.0).

| Setting | As published / original pipeline | Leak-free re-run |
|---|---|---|
| 90:10, alpha=1.0 (paper's Model F), accuracy | 85% (paper); 0.78 in this repo's committed `csv/` | 0.7949 (117 test rows) |
| 90:10, alpha=1.0, Positif (non-hate) precision / recall / F1 | 60 / 16 / 25% (paper, Table 9) | 0.22 / 0.11 / 0.14 |
| 10-fold CV mean accuracy | 79.09% (paper) | 0.8085 (90:10 training portion, alpha=1.0) |
| 80:20, alpha=2.0 (deployed config), accuracy | 0.83 | 0.8170 (235 test rows) |
| 80:20, alpha=2.0, Positif (non-hate) precision / recall / F1 | 0.44 / 0.18 / 0.25 | 0.33 / 0.13 / 0.19 |
| Alpha picked by CV accuracy, per ratio | — | 3.0 for all three ratios; test accuracy 0.8319 / 0.8298 / 0.8205 (70:30 / 80:20 / 90:10), Positif (non-hate) recall 0.09 / 0.11 / 0.11 |

**Majority-class baseline.** Predicting `Negatif` for every row scores 0.8352 / 0.8368 / 0.8333 on the original 70:30 / 80:20 / 90:10 test sets, and 0.8348 / 0.8383 / 0.8376 on the leak-free ones. Every leak-free accuracy above is at or below this baseline. In the original pipeline, alpha=2.0 and alpha=3.0 at 70:30 score exactly the baseline (0.8352), and the deployed config's 0.8285 is below its baseline of 0.8368.

The paper's 85% (90:10, alpha=1.0) and 79.09% CV figures are not reproduced by this repo's current code. The committed results for that setting give 0.78, and a fresh re-run of the original code gives 0.79. The paper's results appear to come from a different run or configuration than the one committed here.

**Label definition.** As defined in the paper (Section 2.1, Table 1): `Positif` = non-hate speech toward Islam, `Negatif` = hate speech against Islam. `dataset.xlsx` has 1,002 `Negatif` (83.5%) and 198 `Positif` (16.5%) rows. Earlier versions of this README described `Positif` as hate speech, which inverted the Key Finding; corrected 2026-11-03.

**scikit-learn version warning.** The pickles were saved with scikit-learn 1.5.1, so loading them under a newer version (the app installs scikit-learn unpinned) raises `InconsistentVersionWarning`. Checked 2026-11-03: under 1.5.1 (Python 3.11) and 1.9.0 (Python 3.14) the deployed model and vectorizer give identical `predict` and `predict_proba` on all 1,200 dataset rows, with identical vocabulary and model parameters, so the warning is benign for these two versions. A future scikit-learn release could still change this.

Full detail: [`ERRATA.md`](ERRATA.md) and [`PROGRESS.md`](PROGRESS.md).

## Publication

Al Juhaeda, Z. I., Faisal, M., & Suhartono. (2024). *Sentiment Classification of Hate Speech Against Islam on Twitter Platform Using Multinomial Naïve Bayes.* Journal of Informatics and Science Media (JISMEDIA), 1(1), 34–38. [View on Garuda](https://garuda.kemdiktisaintek.go.id/documents/detail/4926421)

## Tech Stack

- **Python**, **Jupyter Notebook**
- **scikit-learn** — CountVectorizer, MultinomialNB, KFold cross-validation
- **NLTK** — Indonesian stopword list; stemming uses NLTK's English `PorterStemmer`, applied as-is to Indonesian text (no Indonesian stemmer). Tokenization is `str.split()`; the demo app (`app.py`) calls the same `preprocessing.py` functions as training (until 2026-11-03 it used its own copy with NLTK `word_tokenize` and no URL/mention stripping).
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

- **Small dataset** (~1,200 tweets) for a 2-class problem with real class imbalance — limits how well the minority (`Positif`, non-hate) class can be learned.
- **Single platform, single language variant** — Indonesian-language Twitter text only; informal/slang coverage depends entirely on what's in the training set and lexicon.
- **Bag-of-Words + Naive Bayes** has no notion of word order or context — sarcasm, negation, and indirect hate speech are hard cases this architecture structurally can't catch.
- See **Key Finding** above — low recall on the `Positif` (non-hate) class, i.e. non-hate tweets flagged as hate, is the main limitation, not a footnote.

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
