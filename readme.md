Sentiment Analysis Flask App
This project is a web application that performs sentiment analysis on user-input text. The backend is built using Flask, and it utilizes pre-trained machine learning models to predict whether the sentiment of the text is positive, negative, or neutral.


This project consists of two main parts:

Model Training (train.py)
The machine learning models (Naive Bayes, SVM, and Logistic Regression) were pre-trained using a combined_dataset.csv dataset.
The train.py file is included to demonstrate the training process and the project’s workflow — even though the application uses pre-trained models.

Prediction API and User Interface
The pre-trained models are stored using the joblib library and served through a Flask API.
The web interface allows users to enter text and view sentiment predictions from all three models.

pip install -r requirements.txt
python -m nltk.downloader stopwords punkt
