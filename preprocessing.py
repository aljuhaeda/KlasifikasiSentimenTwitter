# preprocessing.py

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Pastikan NLTK resources telah diunduh
nltk.download('stopwords')

def clean_text(text):
    """
    Membersihkan teks dari URL, mention, hashtag, dan karakter khusus.
    """
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Menghapus URL
    text = re.sub(r'\@\w+|\#','', text)  # Menghapus mention dan hashtag
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Menghapus karakter non-huruf
    return text

def to_lowercase(text):
    """
    Mengonversi teks menjadi huruf kecil.
    """
    return text.lower()

def tokenize_text(text):
    """
    Melakukan tokenisasi pada teks.
    """
    return text.split()

def remove_stopwords(tokens):
    """
    Menghapus stopword dari token.
    """
    stop_words = set(stopwords.words('indonesian'))  # Menggunakan stopword bahasa Indonesia
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

def stem_tokens(tokens):
    """
    Melakukan stemming pada token.
    """
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(word) for word in tokens]
    return stemmed

def preprocess_text(text):
    """
    Melakukan seluruh tahapan preprocessing pada teks.
    """
    cleaned = clean_text(text)
    lower = to_lowercase(cleaned)
    tokens = tokenize_text(lower)
    no_stop = remove_stopwords(tokens)
    stemmed = stem_tokens(no_stop)
    processed = ' '.join(stemmed)
    return cleaned, lower, tokens, no_stop, stemmed, processed
