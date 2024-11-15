# Import necessary libraries
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer and models
# Load the tokenizer
tokenizer = joblib.load('tokenizer2.joblib')

# Load the models
rnn_model = load_model('rnn_sentiment_model5.h5')
lstm_model = load_model('lstm_sentiment_model5.h5')
 
# Streamlit app
st.title("Sentiment Analysis Comparison: RNN and LSTM")
st.write("This app predicts the sentiment (positive or negative) of a given text using a trained RNN and LSTM models.")

# Sidebar model selector
model_choice = st.sidebar.selectbox("Choose a model:", ("RNN", "LSTM"))

# Text input
user_input = st.text_area("Enter a text:")

# Preprocess the input
if user_input:
    sequences = tokenizer.texts_to_sequences([user_input])
    padded_input = pad_sequences(sequences, maxlen=300, padding='post', truncating='post')
    
    # Make prediction based on model selection
    if model_choice == "RNN":
        prediction = rnn_model.predict(padded_input)[0][0]
    else:  
        prediction = lstm_model.predict(padded_input)[0][0]
    # Display results
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    st.write(f"Model: {model_choice}")
    st.write(f"Prediction: {prediction:.2f}")
    st.write(f"Sentiment: {sentiment}")

 
