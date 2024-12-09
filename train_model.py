import pandas as pd
import re
import nltk
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Pastikan Anda telah mengunduh data NLTK sebelumnya
nltk.download('punkt')
nltk.download('stopwords')

# 1. Memuat Dataset
dataset_path = r'C:\SkripsiFix\2. Code\Model\dataset.xlsx'
df = pd.read_excel(dataset_path)

# Tampilkan beberapa baris pertama untuk memastikan data dimuat dengan benar
print("Head of the dataset:")
print(df.head())

# 2. Preprocessing Data
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
    
    return processed_text

# Terapkan preprocessing ke kolom 'full_text'
df['processed_text'] = df['full_text'].apply(preprocess_text)

# 3. Membagi Data
X = df['processed_text']
y = df['kelas']

# Split data menjadi training dan testing set (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Vectorisasi Teks
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# 5. Pelatihan Model
model = MultinomialNB(alpha=1.0)  # Anda dapat mengubah parameter alpha jika diperlukan
model.fit(X_train_counts, y_train)

# 6. Evaluasi Model
y_pred = model.predict(X_test_counts)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(conf_matrix)

# 7. Menyimpan Model dan Vectorizer
model_output_path = r'C:\SkripsiFix\2. Code\Model\pickle\mnb_alpha_1.0_tuning_minimal_ratio_80_20_new.pkl'
vectorizer_output_path = r'C:\SkripsiFix\2. Code\Model\pickle\countvectorizer_new.pkl'

with open(model_output_path, 'wb') as f:
    pickle.dump(model, f)

with open(vectorizer_output_path, 'wb') as f:
    pickle.dump(vectorizer, f)

print(f"Model disimpan di: {model_output_path}")
print(f"Vectorizer disimpan di: {vectorizer_output_path}")
