# KlasifikasiSentimenTwitter — Progress

## Status
Deployed, stable, published research. Live at
https://klasifikasisentimentwitter.streamlit.app (Streamlit Community
Cloud free tier — sleeps after inactivity, wakes on first visit).

## Done
- Multinomial Naive Bayes hate-speech classifier for Indonesian-language
  tweets, with lexicon augmentation and systematic alpha tuning (0.01-3.0)
  across 3 train/test splits, validated with 10-fold CV.
- Published: Al Juhaeda, Z. I. (2024), *Sentiment Analysis of Hate Speech
  in Twitter Comments Using Naïve Bayes Classifier*, JISMAN Journal.
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

## Next up
- Nothing scheduled.
