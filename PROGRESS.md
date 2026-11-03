# KlasifikasiSentimenTwitter — Progress

## Status
Deployed, stable, published research. Live at
https://klasifikasisentimentwitter.streamlit.app (Streamlit Community
Cloud free tier — sleeps after inactivity, wakes on first visit).

## Done
- Multinomial Naive Bayes hate-speech classifier for Indonesian-language
  tweets, with systematic alpha tuning (0.01-3.0) across 3 train/test
  splits, validated with 10-fold CV. (The notebook's lexicon labelling
  step is not used in training; corrected 2026-11-03.)
- Published: Al Juhaeda, Z. I., Faisal, M., & Suhartono (2024),
  *Sentiment Classification of Hate Speech Against Islam on Twitter
  Platform Using Multinomial Naïve Bayes*, Journal of Informatics and
  Science Media (JISMEDIA), Vol. 1 Iss. 1, pp. 34–38, July 2024.
- Deployed config is alpha=2.0 / 80:20 (original pipeline: acc 0.83,
  Positif recall 0.18). The earlier rationale ("trades 1% accuracy for
  +4pp recall over alpha=3.0/70:30") compared different test splits; on
  the same 70:30 test set alpha=2.0 and 3.0 tie at 0.8352 with Positif
  recall 0.20 vs 0.14 (corrected 2026-11-03, see README Results).
- Citation corrected from "JISMAN Journal, Nov 2024" to JISMEDIA Vol. 1
  Iss. 1, pp. 34-38, July 2024 (commit b0be3b5) — closed.
- Leak-free re-evaluation (`eval_leakfree.py` → `results_leakfree.txt`),
  README "Reproducibility note" and `ERRATA.md` added 2026-11-03;
  published figures and original artifacts left unchanged.
- Label meaning corrected repo-wide (`Positif` = non-hate, `Negatif` =
  hate; Key Finding rewritten), app.py preprocessing switched to
  `preprocessing.py` (proven identical to training on all 1,200 rows),
  pickle version warning shown benign — 2026-11-03, see verification log.
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
  well the minority (`Positif`, non-hate) class can be learned.
- Labels (paper §2.1, Table 1): `Positif` = non-hate, `Negatif` = hate
  against Islam (1,002 / 1,200 rows, 83.5%). The deployed config catches
  most hate speech (`Negatif` recall 0.96, precision 0.86 original;
  0.9492 / 0.8500 leak-free) but flags ~4 in 5 non-hate tweets as hate
  (`Positif` recall 0.18 original, 0.1316 leak-free). Accuracy is at or
  below the always-`Negatif` baseline (0.833-0.838). Stated in the
  README's Key Finding. (Earlier wording here called `Positif` the
  hate-speech class; corrected 2026-11-03.)
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
    `app.py` comment "best hate-speech recall among tested configs" (it
    meant `Positif` recall, which is non-hate recall) for
    alpha=2.0/80_20 is wrong: alpha=1.0/80_20 has 0.26, and the sweep
    has 70_30 alpha=0.5 at recall 0.3729.
  - **Unverified / does not reproduce from this repo:** the paper's own
    numbers (Model F 90:10 alpha=1.0, 85% accuracy; 10-fold CV mean
    79.09%, best fold 85.05%; alphas 0.6/0.4; 119 test rows; 1,190
    tweets). This repo's code gives 90_10 alpha=1.0 acc 0.7917 on 120
    test rows. The paper's results came from a different run or
    configuration than the one committed here.
- 2026-11-03 (leak-free re-run): `.venv\Scripts\python.exe
  eval_leakfree.py` → `results_leakfree.txt` (Python 3.14.7, sklearn
  1.9.0, numpy 2.5.2, pandas 3.0.5, nltk 3.10.2; seed 42 for stratified
  split and KFold(10, shuffle)). Run twice, outputs byte-identical.
  Per ratio: split first; `Pipeline(CountVectorizer(max_features=1000),
  MultinomialNB)` fit on train only; test rows whose preprocessed text
  is in train dropped (7/4/3 → 351/235/117 test rows); alpha by 10-fold
  CV accuracy on train over {0.01, 0.1, 0.4, 0.5, 0.6, 0.8, 1.0, 2.0,
  3.0}; test scored once; CV refits the vectorizer per fold.
  - CV-accuracy pick is alpha=3.0 at every ratio: test acc 0.8319 /
    0.8298 / 0.8205 (70:30 / 80:20 / 90:10), Positif recall 0.0862 /
    0.1053 / 0.1053, Positif F1 0.1449 / 0.1667 / 0.1600; CV mean
    0.8295 / 0.8299 / 0.8272.
  - Deployed config (80:20, alpha=2.0) re-fit leak-free: acc 0.8170,
    Positif P/R/F1 0.3333 / 0.1316 / 0.1887, CV mean 0.8278 (original
    pipeline: 0.83 / recall 0.18 / F1 0.25).
  - Paper's Model F (90:10, alpha=1.0) leak-free: acc 0.7949, Positif
    P/R/F1 0.2222 / 0.1053 / 0.1429, CV mean 0.8085 (paper: 85%, Positive
    60/16/25; committed csv 0.78; fresh original-pipeline re-run 0.7917).
  - Secondary (alpha by CV F1(Positif)): 70:30 alpha=0.4 → acc 0.8291,
    Positif recall 0.3103, F1 0.3750; 80:20 alpha=0.5 → 0.7872 / 0.2632 /
    0.2857; 90:10 alpha=0.4 → 0.7692 / 0.1053 / 0.1290.
  - Leak counts re-checked: by row, 238/358 (70:30) and 119/239 (80:20)
    test rows are another ratio's training rows (0/120 at 90:10); by
    text match 241/122/3, matching the earlier entry.
  - Paper arithmetic: the mean of Table 10's ten fold accuracies is
    79.54%, not the stated 79.09%; Table 3 P/R/F1 are support-weighted
    averages (recall = accuracy). Details in `ERRATA.md`.
  - Majority-class baseline (always Negatif): 0.8352 / 0.8368 / 0.8333
    on the original test sets, 0.8348 / 0.8383 / 0.8376 on the leak-free
    ones. No leak-free config beats it on accuracy. Original 70:30
    alpha=2.0/3.0 (0.8352) equals the baseline exactly; the deployed
    0.8285 is below it.
  - CV caveat: 13 / 16 / 18 within-train duplicates after preprocessing
    can split across KFold folds (not grouped), so CV means are slightly
    optimistic.
  - `pickle/countvectorizer.pkl` (loaded by app.py) and
    `pickle/count_vectorizer.pkl` (written by the notebook) have
    identical vocabularies.
  - `openpyxl==3.1.5` installed in `.venv` and added to
    `requirements.txt`; `pd.read_excel('dataset.xlsx')` now loads (1200, 2).
- 2026-11-03 (label fix, app preprocessing, pickle warning):
  - **Labels corrected** across README / PROGRESS / ERRATA /
    `results_leakfree.txt` legend / app.py display text: `Positif` =
    non-hate, `Negatif` = hate against Islam (paper §2.1, Table 1).
    Key Finding rewritten from numbers already in the logged runs and
    `results_leakfree.txt`. `eval_leakfree.py` re-run after the legend
    change: `results_leakfree.txt` differs from the previous version in
    the legend line only (git diff: 1 line).
  - **app.py preprocessing now calls `preprocessing.preprocess_text`.**
    Before: the app's own copy (no URL/mention stripping, `word_tokenize`)
    produced different processed text from training on 982 / 1,200
    dataset rows (e.g. mention handles kept as tokens); deployed-model
    predictions differed on 0 / 1,200 rows, but `predict_proba` was not
    identical. After: app function (extracted from app.py via AST) vs
    `preprocessing.preprocess_text` on all 1,200 rows — cleaned, lower,
    tokens, no-stopword, stemmed and processed text all 0 differing;
    predictions 0 differing; `predict_proba` identical.
    `preprocessing.preprocess_text` reproduces the committed
    `csv/preprocessed_data.csv` `processed_text` on all 1,191 deduped
    rows (0 differing). Headless `streamlit.testing.v1.AppTest` run of
    app.py: no exception, prediction and probability table render.
    Model and pickles unchanged (alpha=2.0 / 80:20).
  - **Pickle `InconsistentVersionWarning` is benign for sklearn 1.9.0.**
    Cause: `mnb_alpha_2.0_tuning_maksimal_ratio_80_20.pkl` and
    `countvectorizer.pkl` were saved with scikit-learn 1.5.1 (warning
    text names 1.5.1 for both); `requirements.txt` leaves scikit-learn
    unpinned. Check: same 1,200 preprocessed texts loaded in a scratch
    venv (Python 3.11.15, scikit-learn 1.5.1, numpy 2.4.6; no version
    warning on load) and in `.venv` (Python 3.14.7, scikit-learn 1.9.0):
    `predict`, `predict_proba`, the BoW matrix, `feature_log_prob_`,
    `class_log_prior_`, `classes_`, `alpha` and `vocabulary_` all
    exactly equal (max abs diff 0.0). Not retrained, pickles not
    regenerated. scikit-learn **not pinned**: 1.5.1 has no wheels for
    Python 3.13+, the repo has no `runtime.txt` / Python setting, and
    Streamlit Cloud's Python for this app can't be seen from here, so a
    `==1.5.1` pin could break the deploy. A future unpinned release could
    still change loading behaviour.
    **Update (2026-11-03, later):** `requirements.txt` now pins
    `scikit-learn==1.9.0` (the version shown above to match the 1.5.1
    pickles exactly) along with the other listed packages. Committed
    locally, not pushed: these pins need Python 3.11 or newer on Streamlit
    Cloud, which can't be checked from here.
  - Close-out: code-reviewer (no blocking issues), security-review (no
    findings; reviewed by hand because the skill's git context ran in
    another directory), simplify (trimmed the now-unused
    `punkt`/`punkt_tab` downloads), verification re-run (equivalence 0
    differing on all fields, AppTest OK, `eval_leakfree.py` exit 0 with
    the legend line as the only diff). Not pushed, so the live demo
    still runs the old preprocessing until the owner pushes.

## Next up
- **Deployed demo config: decided — stays alpha=2.0 / 80:20** (owner
  decision 2026-11-03; model and pickle unchanged). Trade-off on leak-free numbers, where the always-Negatif
  baseline is ~0.835-0.838 and no config beats it on accuracy: the
  CV-accuracy pick (alpha=3.0) gives ~0.82-0.83 accuracy with only
  ~0.09-0.11 Positif (non-hate) recall; the
  current alpha=2.0/80:20 re-fit leak-free gives 0.8170 / 0.1316; a
  recall-oriented pick (70:30, alpha=0.4) gives 0.8291 accuracy with
  0.3103 Positif (non-hate) recall.
- Not done (outside the 2026-11-03 scope): the `requirements.txt` pins
  (scikit-learn 1.9.0 etc.) are committed locally but not pushed. Check
  the app's Python setting in Streamlit Cloud (needs 3.11+), then push and
  confirm the app boots.
