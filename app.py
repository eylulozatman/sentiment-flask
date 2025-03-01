from flask import Flask, request, jsonify, render_template
import joblib
import re
from nltk.corpus import stopwords


app = Flask(__name__)

# Modelleri yükle
svm_model = joblib.load('savedModels/svm_model.pkl')
nb_model = joblib.load('savedModels/Naive_Bayes_model.pkl')
lr_model = joblib.load('savedModels/Logistic_Regression_model.pkl')
vectorizer = joblib.load('savedModels/vectorizer.pkl')

stop_words = set(stopwords.words('english'))

# Metin temizleme fonksiyonu
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)  # URL'leri kaldır
    text = re.sub(r'[^a-zA-Z\s]', '', text)       # Noktalama işaretleri ve sayıları kaldır
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Tahmin fonksiyonu
def predict(text, model):
    cleanText = clean_text(text)
    vectorized = vectorizer.transform([cleanText])
    result = model.predict(vectorized)
    return result[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    text = request.form['text']
    result_svm = predict(text, svm_model)
    result_nb = predict(text, nb_model)
    result_lr = predict(text, lr_model)

    return jsonify({
        'SVM': result_svm,
        'Naive_Bayes': result_nb,
        'Logistic_Regression': result_lr
    })

if __name__ == '__main__':
    app.run(debug=True)
