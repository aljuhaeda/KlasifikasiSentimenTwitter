# Errata and reproducibility notes

The published paper (Al Juhaeda, Faisal & Suhartono, JISMEDIA 1(1), 34–38, July 2024) and the original pipeline's committed results in `csv/` and `pickle/` are unchanged. This file records what a 2026-11-03 re-examination of this repo found, next to the published figures. It is not a retraction.

## 1. Evaluation issues in the original pipeline

| # | Issue | Where | Effect |
|---|---|---|---|
| 1 | `CountVectorizer(max_features=1000)` is fit on the union of the 70:30, 80:20 and 90:10 training sets. 238/358 (70:30) and 119/239 (80:20) test rows are training rows of another ratio. By text match the counts are 241/358, 122/239 and 3/120. | `skripsi.ipynb` cell 12 | Test rows influence the vocabulary (no label leak) |
| 2 | `drop_duplicates()` runs on raw rows only. After preprocessing, 7/358, 4/239 and 3/120 test rows duplicate a row in their own training set. | cell 4 / cell 8 | Slightly optimistic test scores |
| 3 | Alpha is chosen by test-set accuracy (`find_best_alpha` reads the test-accuracy CSVs), and the deployed config was chosen by test recall. | `find_best_alpha.py`, cells 21/23/28 | Model selection on the test set |

## 2. Leak-free re-run

Command: `.venv\Scripts\python.exe eval_leakfree.py`, which writes `results_leakfree.txt`. Seed 42 is used for the stratified `train_test_split` and the `KFold(10, shuffle=True)`; the folds are the same as `kfold_cv.py`. Environment: Python 3.14.7, scikit-learn 1.9.0, numpy 2.5.2, pandas 3.0.5, nltk 3.10.2. Two runs gave byte-identical output.

Method, per ratio: the split is made first. The vectorizer and MNB are fit as one `Pipeline` on the training portion only. Test rows whose preprocessed text appears in the training portion are dropped. Alpha is chosen by 10-fold CV accuracy on the training portion, over the grid 0.01, 0.1, 0.4, 0.5, 0.6, 0.8, 1.0, 2.0, 3.0 (the repo's grid plus the paper's 0.4 and 0.6). The test set is scored once. The CV mean refits the vectorizer inside each fold.

| Ratio | Test rows | CV-accuracy alpha | Test acc | Positif P / R / F1 | Negatif P / R / F1 | CV mean acc |
|---|---|---|---|---|---|---|
| 70:30 | 351 | 3.0 | 0.8319 | 0.4545 / 0.0862 / 0.1449 | 0.8441 / 0.9795 / 0.9068 | 0.8295 |
| 80:20 | 235 | 3.0 | 0.8298 | 0.4000 / 0.1053 / 0.1667 | 0.8489 / 0.9695 / 0.9052 | 0.8299 |
| 90:10 | 117 | 3.0 | 0.8205 | 0.3333 / 0.1053 / 0.1600 | 0.8468 / 0.9592 / 0.8995 | 0.8272 |

Fixed configurations, scored on the same leak-free splits:

| Config | Test acc | Positif P / R / F1 | CV mean acc |
|---|---|---|---|
| 80:20, alpha=2.0 (deployed config, re-fit leak-free) | 0.8170 | 0.3333 / 0.1316 / 0.1887 | 0.8278 |
| 90:10, alpha=1.0 (paper's Model F) | 0.7949 | 0.2222 / 0.1053 / 0.1429 | 0.8085 |
| 70:30, alpha=0.4 (best CV F1(Positif), secondary criterion) | 0.8291 | 0.4737 / 0.3103 / 0.3750 | 0.7863 |

**Majority-class baseline (always `Negatif`).** On the leak-free test sets this scores 0.8348 / 0.8383 / 0.8376 (70:30 / 80:20 / 90:10). On the original test sets it scores 0.8352 / 0.8368 / 0.8333. No leak-free configuration above beats it on accuracy. In the original pipeline, alpha=2.0 and 3.0 at 70:30 (0.8352) equal the baseline, and the deployed 80:20 alpha=2.0 (0.8285) is below it. Accuracy is therefore not informative here; the per-class figures are.

**CV caveat.** 13 / 16 / 18 training rows (70:30 / 80:20 / 90:10) duplicate another training row after preprocessing. Plain `KFold` can place such pairs in different folds, so the CV means are slightly optimistic. Grouped folds were not used, to keep the folds identical to `kfold_cv.py`.

Selecting alpha by CV accuracy pushes it to the top of the grid (3.0) at every ratio, and Positif recall drops to about 0.09–0.11. Selecting by CV F1(Positif) picks 0.4–0.5 instead. At 70:30 that gives Positif recall 0.31 with test accuracy 0.83. At 80:20 it gives recall 0.26 with accuracy 0.79. At 90:10 (alpha=0.4) recall stays at 0.11 and accuracy falls to 0.77, but that test set has only 19 Positif rows.

## 3. Published figures this repo's code does not reproduce

- **Model F (90:10, alpha=1.0): accuracy 85%.** The committed `csv/accuracy_alpha_1.0_tuning_maksimal_ratio_90_10.csv` gives 0.78. A fresh re-run of the original pipeline gives 0.7917 on 120 test rows; the paper reports 119. The leak-free figure is 0.7949.
- **Table 3 precision / recall / F1.** For all six models these match support-weighted averages of the per-class values in Tables 4–9 to within about one point, and recall equals accuracy in every row. The check assumed stratified supports of about 16.5% Positive. For Model F, the class values 60/16/25 (Positive) and 86/98/92 (Negative) with about 19/100 support give 82 / 85 / 81, and recall then equals accuracy. Per-class figures are the ones to compare with the table above.
- **10-fold CV mean 79.09%, best fold 85.05%.** The paper does not state which model or split the CV used. Nine of the ten fold accuracies are whole-number fractions of 107 or 108 rows, which fits 10 folds of a 1,071-row (90:10) training portion; fold 8 (78.38%) fits neither. The arithmetic mean of the ten listed fold accuracies is 79.54%, not 79.09%. The repo's committed `cv_mean_*` CSVs give 0.75–0.80.
- **Dataset size.** The paper reports 1,190 tweets. `dataset.xlsx` has 1,200 rows, which become 1,191 after `drop_duplicates()`.

## 4. Repo claims corrected on 2026-11-03

- "Lexicon augmentation alongside Bag-of-Words features" (README). Cell 10 writes lexicon-derived labels to `*_with_classes.csv`, but training uses the manual labels and BoW features only, so the lexicon never reaches the model.
- "NLTK stemming" (README). Stemming uses NLTK's English `PorterStemmer` on Indonesian text. Training tokenizes with `str.split()`; `app.py` uses `word_tokenize`.
- The deployed-config rationale compared rows from different test splits. On the same 70:30 test set, alpha=2.0 and alpha=3.0 both score 0.8352, and alpha=2.0 has Positif recall 0.20 against 0.14.
- An `app.py` comment said alpha=2.0/80:20 has the "best hate-speech recall among tested configs". It does not: committed results give alpha=1.0/80:20 0.26 and alpha=0.5/70:30 0.36. Only the comment changed; app behaviour is the same.

## 5. Open question for the owner: label meaning

The paper (Section 2.1, Table 1) defines `Positif` as non-hate speech and `Negatif` as hate speech against Islam. The README describes `Positif` as hate speech. In `dataset.xlsx`, `Negatif` is the majority label (995 of 1,191 rows). The README's Key Finding ("misses roughly 4 in 5 real hate-speech tweets") holds only under the README's definition. This file reports numbers per label name and does not resolve the question.
