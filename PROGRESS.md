# KlasifikasiSentimenTwitter — Progress

## Status
Deployed, stable, published research. Live at
https://klasifikasisentimentwitter.streamlit.app (Streamlit Community
Cloud free tier — sleeps after inactivity, wakes on first visit).

## Done
- Multinomial Naive Bayes hate-speech classifier for Indonesian-language
  tweets, with lexicon augmentation and systematic alpha tuning (0.01-3.0)
  across 3 train/test splits, validated with 10-fold CV.
- Published: Al Juhaeda, Z. I., Faisal, M., & Suhartono (2024),
  *Sentiment Classification of Hate Speech Against Islam on Twitter
  Platform Using Multinomial Naïve Bayes*, Journal of Informatics and
  Science Media (JISMEDIA), Vol. 1 Iss. 1, pp. 34–38, July 2024.
- Deployed config deliberately trades 1% accuracy (84% → 83%) for +4pp
  hate-speech recall (0.14 → 0.18) over the highest-accuracy config —
  chosen specifically because that config was the worst at catching
  actual hate speech. Documented as the central finding, not a footnote.
- Buried thesis-machine leftover (`skripsinew.ipynb`, broken/abandoned
  refactor) removed from the repo.
- README explicit about `train_model.py` being a non-reproducible legacy
  script with hardcoded local paths, and that `skripsi.ipynb` carries
  the same local-path pattern in several cells and needs updating for
  a new machine before rerunning.
- Removed a stray 1,882-line `process_log.log` that had been committed to
  the repo, and added it to `.gitignore` (2026-07-31).

## In progress
- Nothing currently active.

## Known issues / honest limitations
- Small dataset (~1,200 tweets) with real class imbalance — caps how
  well the minority (hate-speech) class can be learned.
- The deployed config misses ~4 in 5 real hate-speech tweets (recall
  0.18) — stated plainly in the README's Key Finding section, not
  glossed over.
- Bag-of-Words + Naive Bayes has no notion of word order — structurally
  can't catch sarcasm, negation, or indirect hate speech.
- Free-tier Streamlit hosting sleeps after inactivity (cold start ~20-30s).

## Verification log
- 2026-07-23: git working tree clean, no pending diff. `/security-review`
  skill checked — N/A, diff-based and nothing to review. Live demo woken
  from sleep; `api/v2/app/resume` returned 204 and all core widget JS
  assets loaded with 200s, no console errors.
- 2026-07-31: found and removed the committed `process_log.log` while
  auditing repos for leaked files.
- 2026-08-11: reviewed commits since last check (README screenshot only)
  — no app/model code changed, status unaffected.
- 2026-11-03 (metrics re-reproduced from code on 2026-11-03): ran
  `skripsi.ipynb` cells 4/6/8/12/16/18/21/23 as a scratch script
  (repo-relative paths, nothing written to the repo; the repo's
  `preprocessing.py`/`mnb.py`/`kfold_cv.py` imported as-is), repo
  `.venv` = Python 3.14.7, scikit-learn 1.9.0. `openpyxl` is missing
  from the venv and `requirements.txt`, so `dataset.xlsx` can't be read
  as-is — it was loaded from a scratch-only install. Data: 1,200 rows →
  1,191 after `drop_duplicates()`; stratified `train_test_split`,
  `random_state=42`, test sizes 0.3/0.2/0.1 (358/239/120 test rows).
  Preprocessed text and all three train/test splits match the
  committed CSVs exactly.
  - **Committed artifacts** (committed `mnb_*.pkl` on committed
    `test_bow_*.pkl`, which is also what the app runs):
    alpha=3.0/70_30 acc 0.8352, Positif recall 0.1356, F1 0.2133;
    alpha=2.0/80_20 (deployed) acc 0.8285 / recall 0.1795 / F1 0.2545;
    alpha=1.0/80_20 acc 0.8201 / recall 0.2564 / F1 0.3175 — **the
    README table (0.84/0.14/0.21, 0.83/0.18/0.25, 0.82/0.26/0.32)
    reproduces.** 10-fold CV (`perform_kfold_cv`, KFold shuffle
    `random_state=42`) on the committed train BoW reproduces every
    committed `cv_*` CSV fold-for-fold (mean 0.75/0.77/0.80/0.78/0.80/0.79).
  - **Fresh re-fit from code** does not fully reproduce: the
    `CountVectorizer(max_features=1000)` cut lands on a 219-way
    frequency tie (count 9), and sklearn 1.9.0 picks 54 different words
    than the committed vectorizer. alpha=3.0/70_30 still gives
    0.8352/0.1356/0.2133, but the deployed alpha=2.0/80_20 gives
    acc 0.8243 (0.82, not the claimed 0.83) / recall 0.1795 / F1 0.2500,
    and alpha=1.0/80_20 gives F1 0.3125 (0.31, not 0.32).
  - **Leak check — three real issues, all in the claimed numbers:**
    (1) the vectorizer is fit on the union of all three ratios'
    training texts, and the three splits are different partitions, so
    241/358 (70_30), 122/239 (80_20) and 3/120 (90_10) test rows were
    in the vectorizer's fit corpus (vocabulary only, no labels). Fitting
    on each ratio's own train fold instead gives alpha=3.0/70_30
    0.8324 / recall 0.1017 / F1 0.1667, alpha=2.0/80_20 0.8201 / 0.1538 /
    0.2182, and alpha=1.0/80_20 unchanged at 0.8201 / 0.2564 / 0.3175.
    (2) `drop_duplicates()` runs on raw rows only; after preprocessing,
    7/358, 4/239 and 3/120 test rows have identical text in their own
    training set. (3) alpha is selected on the test set
    (`find_best_alpha` reads test-accuracy CSVs; no validation split),
    and the deployed config was picked by test recall. All reported
    test numbers are therefore optimistic. There's no date column, so
    the time-ordered-split pattern can't be checked.
  - **Claims that don't hold:** the lexicon step (cell 10) writes
    lexicon-derived labels into `*_with_classes.csv`, but training uses
    the manual labels from `train_{ratio}.csv` and BoW features only.
    The lexicon never reaches the model, so the README's "lexicon
    augmentation alongside BoW features" isn't what the code does. The
    `app.py` comment "best hate-speech recall among tested configs" for
    alpha=2.0/80_20 is wrong: alpha=1.0/80_20 has 0.26, and the sweep
    has 70_30 alpha=0.5 at recall 0.3729.
  - **Unverified / does not reproduce from this repo:** the paper's own
    numbers (Model F 90:10 alpha=1.0, 85% accuracy; 10-fold CV mean
    79.09%, best fold 85.05%; alphas 0.6/0.4; 119 test rows; 1,190
    tweets). This repo's code gives 90_10 alpha=1.0 acc 0.7917 on 120
    test rows. The paper's results came from a different run or
    configuration than the one committed here.

## Next up
- Nothing scheduled.
