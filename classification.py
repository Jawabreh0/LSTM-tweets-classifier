import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model and tokenizer
model = load_model('trained-models/tweet_classifier_model.keras')
with open('trained-models/tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

# Function to preprocess new tweets
def preprocess_tweet(tweet, tokenizer, max_length):
    sequence = tokenizer.texts_to_sequences([tweet])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post')
    return padded

# Function to predict the class of a new tweet
def predict_tweet(tweet, model, tokenizer, max_length):
    processed_tweet = preprocess_tweet(tweet, tokenizer, max_length)
    prediction = model.predict(processed_tweet).round().item()
    return "harmful" if prediction == 1 else "normal"

# Example usage
new_tweet = "Fuck You"
result = predict_tweet(new_tweet, model, tokenizer, max_length)
print(f"The tweet is classified as {result}")
