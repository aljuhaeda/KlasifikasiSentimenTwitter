# bow.py

import pickle
from sklearn.feature_extraction.text import CountVectorizer
import logging

def train_vectorizer(all_train_texts, max_features=1000):
    """
    Melatih CountVectorizer pada seluruh data pelatihan.

    Parameters:
    - all_train_texts: list atau Series, seluruh teks pelatihan dari semua split.
    - max_features: int, jumlah fitur maksimum untuk CountVectorizer.

    Returns:
    - vectorizer: CountVectorizer yang telah dilatih.
    """
    vectorizer = CountVectorizer(max_features=max_features)
    vectorizer.fit(all_train_texts)
    return vectorizer

def transform_texts(vectorizer, texts):
    """
    Mengubah teks menjadi matriks fitur Bag of Words.

    Parameters:
    - vectorizer: CountVectorizer yang telah dilatih.
    - texts: list atau Series, teks yang akan diubah.

    Returns:
    - X_bow: sparse matrix, matriks fitur Bag of Words.
    """
    return vectorizer.transform(texts)

def save_vectorizer(vectorizer, pickle_dir):
    """
    Menyimpan objek CountVectorizer.

    Parameters:
    - vectorizer: CountVectorizer yang telah dilatih.
    - pickle_dir: str, path direktori untuk menyimpan file Pickle.
    """
    vectorizer_pkl = f'{pickle_dir}countvectorizer.pkl'
    with open(vectorizer_pkl, 'wb') as f:
        pickle.dump(vectorizer, f)
    logging.info(f'CountVectorizer telah disimpan sebagai "{vectorizer_pkl}".')

def save_bow(X_bow, split_name, pickle_dir, dataset_type='train'):
    """
    Menyimpan matriks fitur Bag of Words.

    Parameters:
    - X_bow: sparse matrix, matriks fitur Bag of Words.
    - split_name: str, nama rasio pembagian data (misalnya, '70_30').
    - pickle_dir: str, path direktori untuk menyimpan file Pickle.
    - dataset_type: str, 'train' atau 'test' untuk tipe dataset.
    """
    bow_pkl = f'{pickle_dir}{dataset_type}_bow_{split_name}.pkl'
    with open(bow_pkl, 'wb') as f:
        pickle.dump(X_bow, f)
    logging.info(f'Matriks Bag of Words untuk {dataset_type} set rasio {split_name} telah disimpan sebagai "{bow_pkl}".')
