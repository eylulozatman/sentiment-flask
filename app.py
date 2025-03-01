from flask import Flask, request, jsonify, render_template
import joblib
import re
from nltk.corpus import stopwords


app = Flask(__name__)

svm_model = joblib.load('savedModels/svm_model.pkl')
nb_model = joblib.load('savedModels/Naive_Bayes_model.pkl')
lr_model = joblib.load('savedModels/Logistic_Regression_model.pkl')
vectorizer = joblib.load('savedModels/vectorizer.pkl')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)  
    text = re.sub(r'[^a-zA-Z\s]', '', text)      
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

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
    if request.is_json:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Text field is required'}), 400
        text = data['text']
    elif 'text' in request.form:
        text = request.form['text']
    else:
        return jsonify({'error': 'Text field is required'}), 400

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
