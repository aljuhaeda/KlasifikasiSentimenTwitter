# eval_leakfree.py
#
# Leak-free re-evaluation (2026-11-03 audit). Does NOT touch pickle/, csv/,
# the deployed model, or the published results. Writes one file:
# results_leakfree.txt
#
# Differences from the original pipeline (skripsi.ipynb):
#   1. CountVectorizer(max_features=1000) is fit on each split's own training
#      portion only (original: fit once on the union of all three ratios'
#      training texts, which includes other ratios' test rows).
#   2. Test rows whose preprocessed text also appears in that split's
#      training portion are removed (original: drop_duplicates on raw rows only).
#   3. alpha is chosen by 10-fold CV on the training portion only
#      (original: chosen by test-set accuracy).
#   4. The 10-fold CV refits the vectorizer inside each fold.
#
# Run:  .venv\Scripts\python.exe eval_leakfree.py

import os
import platform
import sys

import nltk
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from preprocessing import preprocess_text

SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE_DIR, 'results_leakfree.txt')

# Repo grid (csv/accuracy_*.csv) plus the paper-only values 0.4 and 0.6.
ALPHAS = [0.01, 0.1, 0.4, 0.5, 0.6, 0.8, 1.0, 2.0, 3.0]
SPLITS = {'70_30': 0.3, '80_20': 0.2, '90_10': 0.1}
FIXED = [('80_20', 2.0, 'deployed config'), ('90_10', 1.0, "paper's Model F setting")]
LABELS = ['Negatif', 'Positif']

lines = []


def out(s=''):
    print(s)
    lines.append(s)


def make_pipe(alpha=1.0):
    return Pipeline([
        ('vec', CountVectorizer(max_features=1000)),
        ('nb', MultinomialNB(alpha=alpha)),
    ])


def cv10():
    # Same KFold settings as kfold_cv.perform_kfold_cv, so folds are comparable.
    return KFold(n_splits=10, shuffle=True, random_state=SEED)


def per_class(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return {lab: (p[i], r[i], f[i], s[i]) for i, lab in enumerate(LABELS)}


def report_test(name, pipe, X_tr, y_tr, X_te, y_te):
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    acc = accuracy_score(y_te, pred)
    out(f'  {name}: test accuracy {acc:.4f} (n_test={len(y_te)})')
    for lab, (p, r, f, s) in per_class(y_te, pred).items():
        out(f'    {lab:8s} precision {p:.4f}  recall {r:.4f}  F1 {f:.4f}  support {s}')
    cv = cross_val_score(make_pipe(pipe.named_steps['nb'].alpha), X_tr, y_tr, cv=cv10(), scoring='accuracy')
    out(f'    10-fold CV on training portion (vectorizer fit inside each fold): '
        f'mean accuracy {cv.mean():.4f} (std {cv.std():.4f})')


def main():
    np.random.seed(SEED)
    out('Leak-free re-evaluation, KlasifikasiSentimenTwitter')
    out(f'python {platform.python_version()}, scikit-learn {sklearn.__version__}, '
        f'numpy {np.__version__}, pandas {pd.__version__}, nltk {nltk.__version__}')
    out(f'seed {SEED} (train_test_split stratified, KFold shuffle); alpha grid {ALPHAS}')
    out('Class names are the dataset labels (Positif / Negatif); see README for what each label means.')
    out()

    df = pd.read_excel(os.path.join(BASE_DIR, 'dataset.xlsx'))
    n_raw = len(df)
    df = df.drop_duplicates()
    df['processed_text'] = df['full_text'].apply(lambda t: preprocess_text(t)[5])
    out(f'rows: {n_raw} raw -> {len(df)} after drop_duplicates(); '
        f'class counts {df["kelas"].value_counts().to_dict()}')
    out()

    for ratio, test_size in SPLITS.items():
        X_tr, X_te, y_tr, y_te = train_test_split(
            df['processed_text'], df['kelas'],
            test_size=test_size, stratify=df['kelas'], random_state=SEED)
        train_texts = set(X_tr)
        keep = ~X_te.isin(train_texts)
        n_dup = int((~keep).sum())
        base_orig = (y_te == 'Negatif').mean()
        X_te, y_te = X_te[keep], y_te[keep]
        out(f'=== ratio {ratio}: train {len(X_tr)}, test {len(keep)} -> {len(X_te)} '
            f'after removing {n_dup} test rows duplicating a training row (post-preprocessing)')
        out(f'  majority-class baseline (always Negatif): {base_orig:.4f} on the original '
            f'{len(keep)}-row test set, {(y_te == "Negatif").mean():.4f} on the {len(X_te)}-row leak-free test set')
        n_dup_tr = int(X_tr.duplicated().sum())
        out(f'  caveat: {n_dup_tr} training rows duplicate another training row after preprocessing; '
            f'these can fall in different CV folds (plain KFold, not grouped)')

        # F1 for the Positif label needs an explicit pos_label with string labels.
        grid = GridSearchCV(
            make_pipe(), {'nb__alpha': ALPHAS}, cv=cv10(),
            scoring={'accuracy': 'accuracy',
                     'f1_positif': make_scorer(f1_score, pos_label='Positif', zero_division=0)},
            refit=False)
        grid.fit(X_tr, y_tr)
        res = grid.cv_results_
        out('  CV on training portion only, per alpha (mean accuracy / mean F1 Positif):')
        for i, a in enumerate(res['param_nb__alpha']):
            out(f'    alpha {float(a):<5}: {res["mean_test_accuracy"][i]:.4f} / {res["mean_test_f1_positif"][i]:.4f}')
        best_acc = float(res['param_nb__alpha'][int(np.argmax(res['mean_test_accuracy']))])
        best_f1 = float(res['param_nb__alpha'][int(np.argmax(res['mean_test_f1_positif']))])
        out(f'  alpha chosen by CV accuracy (primary, same criterion as original): {best_acc}')
        out(f'  alpha chosen by CV F1(Positif) (secondary): {best_f1}')
        report_test(f'alpha={best_acc} [CV-accuracy pick]', make_pipe(best_acc), X_tr, y_tr, X_te, y_te)
        if best_f1 != best_acc:
            report_test(f'alpha={best_f1} [CV-F1(Positif) pick]', make_pipe(best_f1), X_tr, y_tr, X_te, y_te)
        for f_ratio, f_alpha, label in FIXED:
            if f_ratio == ratio:
                report_test(f'alpha={f_alpha} [fixed: {label}]', make_pipe(f_alpha), X_tr, y_tr, X_te, y_te)
        out()

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'written: {OUT_PATH}', file=sys.stderr)


if __name__ == '__main__':
    main()
