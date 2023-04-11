import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from flask import Flask
app = Flask(__name__)
model = load_model('sentiment_analysis_rnn.h5')
with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)
def predict_sentiment(text): 
    processed_text = preprocess(text)
    tokenized_text = tokenizer.texts_to_sequences([processed_text])
    padded_text = pad_sequences(tokenized_text, maxlen=max_len)
    prediction = model.predict(padded_text)
    if prediction >= 0.5:
        label = 'positive'
    else:
        label = 'negative'

    return label


@app.route('/')
def home():
    return render_template('index.html')

  @app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment_api():
    if request.method == 'POST':
        text = request.form['text']
        label = predict_sentiment(text)
        return jsonify({'sentiment': label})
if __name__ == '__main__':
    app.run()
