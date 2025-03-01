import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

nltk.download('stopwords')
nltk.download('punkt')

DATA_PATH = "set/combined_dataset.csv"

def load_data(file_path):
    df = pd.read_csv(file_path, quotechar='"', encoding='utf-8')
    df.fillna("", inplace=True)
    return df

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)     

    stop_words = set(stopwords.words('english'))
    important_words = {'not', 'wasn', 'isn', 'couldn', 'wouldn', 'won', 'don'}
    stop_words = stop_words - important_words

    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


def vectorize_text(train_texts, test_texts):
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer

def main():
    df = load_data(DATA_PATH)
    df['clean_text'] = df['text'].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['sentiment'], test_size=0.2, random_state=42)
    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)

    models = {
        'Naive Bayes': MultinomialNB(),
        'SVM': SVC(kernel='linear'),
        'Logistic Regression': LogisticRegression(max_iter=5000)
    }

    joblib.dump(vectorizer, 'savedModels/vectorizer.pkl')
    print("Vectorizer başarıyla kaydedildi.")

    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        print(f'\nModel: {name}')
        print('Accuracy:', acc)
        print(classification_report(y_test, y_pred))

        model_filename = f'savedModels/{name.replace(" ", "_")}_model.pkl'
        joblib.dump(model, model_filename)
        print(f"{name} başarıyla kaydedildi.")

if __name__ == "__main__":
    main()
