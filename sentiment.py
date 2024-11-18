# streamlit_app.py

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

# Load the models
lstm_model = tf.keras.models.load_model('yelp_sentiment_lstm_model.h5')
rnn_model = tf.keras.models.load_model('rnn_sentiment_model5.h5')

# Load the tokenizer
tokenizer = joblib.load('tokenizer.joblib')

# Add CSS for styling
st.markdown("""
    <style>
    .title {
        font-size: 40px;
        color: #4CAF58;
        text-align: center;
    }
    .text-area {
        font-size: 18px;
        color: #2C3E50;
    }
    .result {
        font-size: 24px;
        color: #4CAF58;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app title and introduction
st.markdown('<h1 class="title">Sentiment Analysis: RNN vs LSTM</h1>', unsafe_allow_html=True)
st.write("Analyze the sentiment (positive or negative) of your text using RNN or LSTM-based models. Compare their performance and results.")

# Model selection
model_choice = st.radio("Choose a model for prediction:", ["RNN", "LSTM"])

# Text input from the user
st.markdown('<p class="text-area">Enter your text below:</p>', unsafe_allow_html=True)
user_input = st.text_area("")

# Preprocess and predict sentiment
if st.button("Analyze Sentiment"):
    if user_input:
        # Preprocess the input text
        input_sequence = tokenizer.texts_to_sequences([user_input])
        input_padded = pad_sequences(input_sequence, maxlen=300, padding='post', truncating='post')

        # Select the model
        if model_choice == "RNN":
            prediction = rnn_model.predict(input_padded)[0][0]
        else:  # LSTM
            prediction = lstm_model.predict(input_padded)[0][0]

        # Display the result
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        confidence = prediction if sentiment == "Positive" else 1 - prediction
        st.markdown(f'<p class="result"><strong>Sentiment:</strong> {sentiment}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="result"><strong>Confidence:</strong> {confidence:.2f}</p>', unsafe_allow_html=True)
    else:
        st.write("Please enter text for analysis.")

 
