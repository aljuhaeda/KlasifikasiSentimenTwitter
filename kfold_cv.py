# kfold_cv.py

import pickle
import logging
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

def load_model(pickle_path):
    """
    Memuat model dari file Pickle.

    Parameters:
    - pickle_path: str, path ke file Pickle model.

    Returns:
    - model: model yang telah dimuat.
    """
    with open(pickle_path, 'rb') as f:
        model = pickle.load(f)
    logging.info(f'Model telah dimuat dari "{pickle_path}".')
    return model

def perform_kfold_cv(model, X, y, n_splits=10):
    """
    Melakukan K-Fold Cross-Validation pada model.

    Parameters:
    - model: model yang akan dievaluasi.
    - X: sparse matrix, fitur.
    - y: array-like, label.
    - n_splits: int, jumlah fold (default=10).

    Returns:
    - metrics_df: DataFrame, metrik evaluasi per fold.
    - accuracies: list of float, skor akurasi untuk setiap fold.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = []

    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Melatih model pada fold ini
        model.fit(X_train, y_train)
        
        # Memprediksi pada fold ini
        y_pred = model.predict(X_test)
        
        # Menghitung metrik
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        # Menyimpan metrik
        metrics.append({
            'fold': fold,
            'test_accuracy': round(accuracy, 2),
            'test_precision_macro': round(precision_macro, 2),
            'test_recall_macro': round(recall_macro, 2),
            'test_f1_macro': round(f1_macro, 2)
        })
        
        logging.info(f'Fold {fold}: Accuracy={accuracy:.4f}, Precision_macro={precision_macro:.4f}, Recall_macro={recall_macro:.4f}, F1_macro={f1_macro:.4f}')
    
    metrics_df = pd.DataFrame(metrics)
    accuracies = [m['test_accuracy'] for m in metrics]
    return metrics_df, accuracies

def save_cv_results(metrics_df, accuracies, ratio_name, alpha, tuning_type, csv_dir):
    """
    Menyimpan hasil cross-validation ke dalam file CSV.

    Parameters:
    - metrics_df: DataFrame, metrik evaluasi per fold.
    - accuracies: list of float, skor akurasi untuk setiap fold.
    - ratio_name: str, nama rasio pembagian data.
    - alpha: float, nilai hyperparameter alpha.
    - tuning_type: str, jenis tuning ('minimal' atau 'maksimal').
    - csv_dir: str, path direktori untuk menyimpan file CSV.
    """
    # Menyimpan metrik per fold
    cv_metrics_csv = f'{csv_dir}cv_metrics_alpha_{alpha}_tuning_{tuning_type}_ratio_{ratio_name}.csv'
    metrics_df.to_csv(cv_metrics_csv, index=False)
    logging.info(f'CV Metrik disimpan sebagai "{cv_metrics_csv}".')
    
    # Menyimpan akurasi per fold
    cv_accuracy_df = pd.DataFrame({
        'test_accuracy': metrics_df['test_accuracy']
    })
    cv_accuracy_csv = f'{csv_dir}cv_accuracy_alpha_{alpha}_tuning_{tuning_type}_ratio_{ratio_name}.csv'
    cv_accuracy_df.to_csv(cv_accuracy_csv, index=False)
    logging.info(f'CV Akurasi disimpan sebagai "{cv_accuracy_csv}".')
    
    # Menyimpan rata-rata akurasi
    mean_accuracy = round(sum(accuracies) / len(accuracies), 2)
    cv_mean_accuracy_df = pd.DataFrame({
        'mean_accuracy': [mean_accuracy]
    })
    cv_mean_accuracy_csv = f'{csv_dir}cv_mean_accuracy_alpha_{alpha}_tuning_{tuning_type}_ratio_{ratio_name}.csv'
    cv_mean_accuracy_df.to_csv(cv_mean_accuracy_csv, index=False)
    logging.info(f'CV Rata-rata Akurasi disimpan sebagai "{cv_mean_accuracy_csv}".')
