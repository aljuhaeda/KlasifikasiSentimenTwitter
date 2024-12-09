# mnb.py

import pickle
import logging
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def train_mnb(X_train, y_train, alpha=1.0):
    """
    Melatih model Multinomial Naive Bayes dengan hyperparameter alpha tertentu.
    
    Parameters:
    - X_train: sparse matrix, matriks fitur Bag of Words untuk training set.
    - y_train: Series atau array, label kelas untuk training set.
    - alpha: float, smoothing parameter.
    
    Returns:
    - model: MultinomialNB, model yang telah dilatih.
    """
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Mengevaluasi model dengan menghitung akurasi dan menghasilkan laporan klasifikasi.
    
    Parameters:
    - model: MultinomialNB, model yang telah dilatih.
    - X_test: sparse matrix, matriks fitur Bag of Words untuk testing set.
    - y_test: Series atau array, label kelas untuk testing set.
    
    Returns:
    - accuracy: float, skor akurasi model.
    - report: dict, laporan klasifikasi.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return accuracy, report

def save_model(model, ratio_name, alpha, tuning_type, pickle_dir):
    """
    Menyimpan model yang telah dilatih.
    
    Parameters:
    - model: MultinomialNB, model yang telah dilatih.
    - ratio_name: str, nama rasio pembagian data (misalnya, '70_30').
    - alpha: float, nilai hyperparameter alpha yang digunakan.
    - tuning_type: str, jenis tuning ('minimal' atau 'maksimal').
    - pickle_dir: str, path direktori untuk menyimpan file Pickle.
    """
    model_pkl = f'{pickle_dir}mnb_alpha_{alpha}_tuning_{tuning_type}_ratio_{ratio_name}.pkl'
    with open(model_pkl, 'wb') as f:
        pickle.dump(model, f)
    logging.info(f'Model MNB dengan alpha={alpha}, tuning={tuning_type} untuk rasio {ratio_name} telah disimpan sebagai "{model_pkl}".')

def save_evaluation(accuracy, report, ratio_name, alpha, tuning_type, csv_dir):
    """
    Menyimpan hasil evaluasi model ke dalam file CSV.
    
    Parameters:
    - accuracy: float, skor akurasi model.
    - report: dict, laporan klasifikasi.
    - ratio_name: str, nama rasio pembagian data (misalnya, '70_30').
    - alpha: float, nilai hyperparameter alpha yang digunakan.
    - tuning_type: str, jenis tuning ('minimal' atau 'maksimal').
    - csv_dir: str, path direktori untuk menyimpan file CSV.
    """
    import pandas as pd

    # Membuat DataFrame untuk akurasi
    accuracy_df = pd.DataFrame({
        'ratio': [ratio_name],
        'alpha': [alpha],
        'accuracy': [round(accuracy, 2)]
    })

    # Menyimpan akurasi
    accuracy_csv = f'{csv_dir}accuracy_alpha_{alpha}_tuning_{tuning_type}_ratio_{ratio_name}.csv'
    accuracy_df.to_csv(accuracy_csv, index=False)
    logging.info(f'Akurasi untuk alpha={alpha}, tuning={tuning_type}, dan rasio {ratio_name} disimpan sebagai "{accuracy_csv}".')

    # Membuat DataFrame untuk laporan klasifikasi tanpa 'support'
    report_df = pd.DataFrame(report).transpose().drop(columns=['support'])
    report_df = report_df.round(2)  # Membulatkan angka ke dua decimal

    report_csv = f'{csv_dir}classification_report_alpha_{alpha}_tuning_{tuning_type}_ratio_{ratio_name}.csv'
    report_df.to_csv(report_csv)
    logging.info(f'Laporan klasifikasi untuk alpha={alpha}, tuning={tuning_type}, dan rasio {ratio_name} disimpan sebagai "{report_csv}".')
