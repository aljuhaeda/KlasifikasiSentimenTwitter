import streamlit as st
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Mengatur konfigurasi halaman
st.set_page_config(
    page_title="🐦 Klasifikasi Sentimen Ujaran Kebencian",
    page_icon="🐦",
    layout="wide",  # Menggunakan layout wide untuk ruang lebih
    initial_sidebar_state="auto",
)

# Menambahkan logo di sidebar (opsional)
# Jika Anda memiliki file logo, misalnya 'logo.png', letakkan di direktori 'assets/' dan gunakan kode berikut.
logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'logo.png')  # Path relatif
try:
    logo = open(logo_path, "rb").read()
    st.sidebar.image(logo, use_column_width=True)
except FileNotFoundError:
    pass  # Tidak menampilkan logo jika file tidak ditemukan

# Judul Proyek
st.title("🐦 Klasifikasi Sentimen Ujaran Kebencian Terhadap Agama Islam Pada Platform Twitter Menggunakan Multinomial Naive Bayes")

# Deskripsi Singkat
st.markdown("""
Aplikasi web sederhana untuk mengklasifikasikan teks ujaran kebencian terhadap agama Islam pada platform Twitter menggunakan metode Multinomial Naive Bayes, oleh Zul Iflah Al Juhaeda. 📝
""")

# Fungsi untuk memuat model dengan caching_resource
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model file not found at path: {path}")
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

# Fungsi untuk memuat vectorizer dengan caching_resource
@st.cache_resource
def load_vectorizer(path):
    if not os.path.exists(path):
        st.error(f"Vectorizer file not found at path: {path}")
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

# Fungsi Preprocessing
def preprocess_text(text):
    # Cleaning: Menghapus karakter non-alphanumerik
    text_clean = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Case Folding: Mengubah semua huruf menjadi huruf kecil
    text_lower = text_clean.lower()
    
    # Tokenizing: Memisahkan teks menjadi kata-kata
    tokens = word_tokenize(text_lower)
    
    # Stopword Removal: Menghapus kata-kata umum yang tidak berpengaruh
    stop_words = set(stopwords.words('indonesian'))  # Sesuaikan dengan bahasa yang digunakan
    tokens_no_stop = [word for word in tokens if word not in stop_words]
    
    # Stemming: Mengubah kata ke bentuk dasarnya
    stemmer = PorterStemmer()
    tokens_stemmed = [stemmer.stem(word) for word in tokens_no_stop]
    
    # Menggabungkan kembali menjadi string
    processed_text = ' '.join(tokens_stemmed)
    
    return {
        "Original Text": text,
        "Cleaned Text": text_clean,
        "Lowercase Text": text_lower,
        "Tokens": tokens,
        "Tokens without Stopwords": tokens_no_stop,
        "Stemmed Tokens": tokens_stemmed,
        "Processed Text": processed_text
    }

# Fungsi untuk mendapatkan kata kunci yang berpengaruh menggunakan feature_log_prob_
def get_top_features(model, vectorizer, class_label, top_n=10):
    feature_names = vectorizer.get_feature_names_out()
    class_index = list(model.classes_).index(class_label)
    # feature_log_prob_ memberikan log probabilitas kata untuk setiap kelas
    topn = sorted(zip(model.feature_log_prob_[class_index], feature_names), reverse=True)[:top_n]
    return topn

# Build absolute paths
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'pickle', 'mnb_alpha_1.0_tuning_minimal_ratio_80_20.pkl')  # Sesuaikan dengan nama file model Anda
vectorizer_path = os.path.join(base_dir, 'pickle', 'countvectorizer.pkl')  # Sesuaikan dengan nama file vectorizer Anda

# Area Input Teks dan Tombol Predict menggunakan layout columns
col1, col2 = st.columns([3, 1])  # Membuat dua kolom, kolom pertama lebih lebar

with col1:
    input_text = st.text_area(
        "Masukkan Teks untuk Diklasifikasikan:",
        height=150,
        placeholder="Masukan teks yang akan diklasifikasikan disini",
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)  # Menambahkan jarak
    predict_button = st.button("Predict")

# Tombol Predict dan proses prediksi
if predict_button:
    if input_text.strip() == "":
        st.error("Input teks tidak boleh kosong.")
    elif len(input_text.strip().split()) < 3:
        st.warning("Masukkan setidaknya 3 kata untuk klasifikasi yang lebih akurat.")
    else:
        # Memuat model dan vectorizer
        model = load_model(model_path)
        vectorizer = load_vectorizer(vectorizer_path)
        
        # Pastikan model dan vectorizer berhasil dimuat
        if model is not None and vectorizer is not None:
            # Preprocessing teks input
            preprocessing_result = preprocess_text(input_text)
            original_text = preprocessing_result["Original Text"]
            cleaned_text = preprocessing_result["Cleaned Text"]
            lower_text = preprocessing_result["Lowercase Text"]
            tokens = preprocessing_result["Tokens"]
            tokens_no_stop = preprocessing_result["Tokens without Stopwords"]
            tokens_stemmed = preprocessing_result["Stemmed Tokens"]
            processed_text = preprocessing_result["Processed Text"]
            
            # Konversi teks ke BoW menggunakan vectorizer
            X_input = vectorizer.transform([processed_text])
            
            # Prediksi kelas
            prediction = model.predict(X_input)[0]
            
            # Probabilitas prediksi
            probabilities = model.predict_proba(X_input)[0]
            prob_df = pd.DataFrame({
                'Kelas': model.classes_,
                'Probabilitas': probabilities
            })
            
            # Menampilkan hasil prediksi dan probabilitas dalam dua kolom
            result_col1, result_col2 = st.columns([2, 1])
            
            with result_col1:
                st.success(f"Hasil Prediksi: **{prediction}**")
                st.markdown("### **Probabilitas Prediksi:**")
                st.write(prob_df)
            
            # Menghilangkan tombol unduh
            
            # Dropdown Proses Klasifikasi
            with st.expander("🔍 Proses Klasifikasi"):
            
                # Menampilkan hasil preprocessing
                st.markdown("### **Hasil Preprocessing:**")
                st.write(f"**Cleaning:** {cleaned_text}")
                st.write(f"**Case Folding:** {lower_text}")
                st.write("**Tokenizing:**")
                st.write(tokens)
                st.write("**Remove Stopwords:**")
                st.write(tokens_no_stop)
                st.write("**Stemming:**")
                st.write(tokens_stemmed)
                st.write(f"**Processed Text:** {processed_text}")
                
                # Menampilkan Bag of Words (BoW) tanpa nilai nol
                st.markdown("### **Bag of Words (BoW):**")
                st.write("Representasi numerik teks setelah CountVectorizer (hanya fitur dengan nilai tidak nol):")
                # Mengambil fitur dengan nilai tidak nol
                bow_array = X_input.toarray()[0]
                non_zero_indices = bow_array != 0
                bow_features = bow_array[non_zero_indices]
                bow_feature_names = vectorizer.get_feature_names_out()[non_zero_indices]
                bow_df = pd.DataFrame({
                    'Fitur': bow_feature_names,
                    'Nilai': bow_features
                })
                st.write(bow_df)
                
                # Menampilkan Multinomial Naive Bayes (MNB) Tuning dan Validasi Silang
                st.markdown("### **Multinomial Naive Bayes (MNB):**")
                st.write("- **Tuning Model:**")
                st.write("Parameter `alpha` pada Multinomial Naive Bayes diubah untuk mengontrol smoothing dan menghindari overfitting.")
                st.write("- **Validasi Silang (K-Fold Cross-Validation):")
                st.write("Metode untuk mengevaluasi performa model dengan membagi data menjadi beberapa subset (fold).")
        
    # Pembatas
    st.markdown("---")
    
    # Penjelasan Model, Dataset, dan Pentingnya Klasifikasi
    st.header("📚 Penjelasan Model dan Dataset")
    
    st.subheader("Model: Multinomial Naive Bayes")
    st.write("""
    Multinomial Naive Bayes adalah salah satu algoritma machine learning yang efektif untuk klasifikasi teks, terutama dalam konteks pengklasifikasian sentimen. Algoritma ini bekerja dengan menghitung probabilitas setiap kelas berdasarkan fitur yang ada.
    """)
    
    st.subheader("Dataset")
    st.write("""
    Dataset yang digunakan terdiri dari sejumlah tweet yang dikumpulkan dari platform Twitter, masing-masing dilabeli sebagai 'Positif' atau 'Negatif'.
    """)
    
    st.subheader("Pentingnya Klasifikasi Sentimen")
    st.write("""
    Klasifikasi sentimen terhadap ujaran kebencian sangat penting untuk memahami dan menangani potensi ancaman sosial. Dengan kemampuan untuk secara otomatis mengidentifikasi dan mengkategorikan ujaran kebencian, platform dapat lebih efektif dalam mengelola konten dan menjaga lingkungan online yang sehat.
    """)
