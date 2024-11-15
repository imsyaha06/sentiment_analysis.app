# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

# Load the model
model = tf.keras.models.load_model('rnn_sentiment_model5.h5')
 

# Load the tokenizer
tokenizer = joblib.load('tokenizer2.joblib')


# Streamlit app title and introduction
st.title("Sentiment Analysis with RNN")
st.write("This app predicts the sentiment (positive or negative) of a given text using a trained Simple RNN model.")

# Text input from the user
user_input = st.text_area("Enter your text here:")

# Preprocess and predict sentiment
if st.button("Analyze Sentiment"):
    if user_input:
        # Preprocess the input text
        input_sequence = tokenizer.texts_to_sequences([user_input])
        input_padded = pad_sequences(input_sequence, maxlen=300, padding='post', truncating='post')

        # Make prediction
        prediction = model.predict(input_padded)[0][0]

        # Display the result
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        confidence = prediction if sentiment == "Positive" else 1 - prediction
        st.write(f"**Sentiment**: {sentiment}")
        st.write(f"**Confidence**: {confidence:.2f}")
    else:
        st.write("Please enter text for analysis.")
