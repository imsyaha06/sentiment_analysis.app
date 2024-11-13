# import streamlit as st
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import pickle

# # Load the saved model
# model = load_model('lstm_sentiment_model5.h5')

# # Load the saved tokenizer
# with open('tokenizer2.pkl', 'rb') as handle:
#     tokenizer = pickle.load(handle)

# # Streamlit app
# st.title("Sentiment Analysis on Yelp Reviews")
# st.write("This app predicts whether a given sentence has a positive or negative sentiment based on Yelp reviews.")

# # Input text
# user_input = st.text_input("Enter a sentence for sentiment analysis:")

# # Prediction function
# def predict_sentiment(text):
#     # Tokenize and pad the input sentence
#     sequence = tokenizer.texts_to_sequences([text])
#     padded_sequence = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')
    
#     # Make the prediction
#     prediction = model.predict(padded_sequence)
#     return "Positive" if prediction > 0.5 else "Negative"

# # Display prediction
# if user_input:
#     sentiment = predict_sentiment(user_input)
#     st.write(f"Sentiment: {sentiment}")














# import streamlit as st
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, classification_report

# # Load the trained model
# model = tf.keras.models.load_model('lstm_sentiment_model5.h5')

# # Initialize Tokenizer with the same configuration as during training
# tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
# # Assume we have access to the training data used to fit the tokenizer initially
# # Here, we're simulating tokenizer fitting with sample text to load real tokenization parameters.
# train_data_sample = ["Sample text to initialize tokenizer. Replace this with actual training text."]
# tokenizer.fit_on_texts(train_data_sample)

# # Define function to preprocess and predict sentiment
# def preprocess_and_predict(text):
#     # Tokenize and pad the input text
#     sequence = tokenizer.texts_to_sequences([text])
#     padded_sequence = pad_sequences(sequence, maxlen=300, padding='post', truncating='post')
    
#     # Predict sentiment
#     prediction = model.predict(padded_sequence)[0][0]
#     sentiment = "Positive" if prediction > 0.5 else "Negative"
#     confidence = prediction if sentiment == "Positive" else 1 - prediction
#     return sentiment, confidence

# # Streamlit app layout
# st.title("Sentiment Analysis App")
# st.write("Enter a review or statement below to predict its sentiment.")

# # Text input for user
# user_input = st.text_area("Review Text", "Type here...")

# if st.button("Analyze"):
#     if user_input:
#         sentiment, confidence = preprocess_and_predict(user_input)
#         st.write(f"**Predicted Sentiment:** {sentiment}")
#         st.write(f"**Confidence:** {confidence:.2f}")
#     else:
#         st.write("Please enter a review text.")

# # # Optional: Display a confusion matrix for the model's performance on test data
# # if st.checkbox("Show Confusion Matrix"):
# #     # Test data should be loaded or simulated here if not available in Streamlit
# #     # Assuming we have `test_padded` and `test_labels` from test set preparation
# #     predictions = (model.predict(test_padded) > 0.5).astype(int)
# #     confusion_mtx = confusion_matrix(test_labels, predictions)
# #     fig, ax = plt.subplots(figsize=(8, 6))
# #     sns.heatmap(confusion_mtx, annot=True, fmt="d", cmap="Blues", ax=ax,
# #                 xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
# #     ax.set_xlabel("Predicted")
# #     ax.set_ylabel("Actual")
# #     st.pyplot(fig)














# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the model
model = tf.keras.models.load_model('rnn_sentiment_model5.h5')

# Load the tokenizer
with open('tokenizer2.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

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

 