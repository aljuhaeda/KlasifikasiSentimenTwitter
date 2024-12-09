# find_best_alpha.py

import os
import pandas as pd

def find_best_alpha(ratio, tuning_type, alphas, csv_dir):
    """
    Menentukan alpha terbaik berdasarkan akurasi tertinggi.

    Parameters:
    - ratio: str, nama rasio pembagian data (misalnya, '70_30').
    - tuning_type: str, jenis tuning ('minimal' atau 'maksimal').
    - alphas: list of float, daftar nilai alpha yang telah diuji.
    - csv_dir: str, path direktori dimana file CSV disimpan.

    Returns:
    - best_alpha: float, alpha dengan akurasi tertinggi.
    """
    best_alpha = None
    best_accuracy = -1.0

    for alpha in alphas:
        accuracy_csv = f'{csv_dir}accuracy_alpha_{alpha}_tuning_{tuning_type}_ratio_{ratio}.csv'
        if os.path.exists(accuracy_csv):
            df = pd.read_csv(accuracy_csv)
            current_accuracy = df['accuracy'].iloc[0]
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_alpha = alpha
        else:
            print(f'Warning: {accuracy_csv} tidak ditemukan.')

    return best_alpha
