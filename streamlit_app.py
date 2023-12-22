# Streamlit app

# Relevant libraries
import re, os, time
from flask import Flask, request, jsonify, send_file
import tensorflow as tf
import numpy as np

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('snowball_data')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

# Streamlit
import streamlit as st

# Set page title, sidebar title, icon and comments
st.title('Sentiment140 Predicting tweets')

st.sidebar.title("Context")

st.sidebar.image("image/air_paradis.png", width=200)

st.sidebar.write(":red[Air Paradis wants an AI prototype to predict sentiment from tweets]. \n \n Sentiment140 dataset was used to create my model. You can find the dataset [here](https://www.kaggle.com/datasets/kazanova/sentiment140).\n \n The model is a LSTM unidirectional Word2Vec embedding on raw text. \n \n :green[You'll be returned positive if score is above 0.5 and negative if score is below 0.5]. \n \n :green[0 means very negative tweet and 1 for very positive tweet]. \n \n Let's try it! \n \n :blue[C.C]")

# Maximum length sequence used during training
MAX_SEQUENCE_LENGTH = 140

# Load stopwords
stop_words = stopwords.words('english')

# Cleaning pattern
text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

# Load the model and the tokenizer
with st.spinner('Loading classification model...'):
    classification_model = load_model('model.h5')
    
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
print(classification_model.summary())

def preprocess(text, stem_or_lem="lem"):
  text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
  tokens = []
  for token in text.split():
    if token not in stop_words:
      if stem_or_lem == "stem":
        stemmer = SnowballStemmer('english')
        tokens.append(stemmer.stem(token))
      else:
        lemmatizer = WordNetLemmatizer()
        tokens.append(lemmatizer.lemmatize(token))
  return " ".join(tokens)

st.subheader('Tweet classification')

def predict_sentiment(text):
    # Preprocess the text in the same way than for the training
    text = preprocess(text)

    # Get the index sequences from the tokenizer
    index_sequence = pad_sequences(tokenizer.texts_to_sequences([text]),
                                   maxlen=MAX_SEQUENCE_LENGTH)

    probability_score = classification_model.predict(index_sequence)[0][0]

    if probability_score < 0.5:
        sentiment = "negative"
    else:
        sentiment = "positive"

    return sentiment, probability_score

text = st.text_input('Tweet:')

if text != '':
    results = predict_sentiment(text)
    with st.spinner('Predicting...'):
        st.write('Prediction:')
        st.write(results[0] + ' with ',
                round(results[1],3),'score')
