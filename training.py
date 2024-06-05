import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

# Load datasets
train_normal = pd.read_csv('/Users/cypruscodes/Desktop/Bandar_Project/ltsm-tweets-classifier/dataset/training_dataset/normal.csv')
train_harmful = pd.read_csv('/Users/cypruscodes/Desktop/Bandar_Project/ltsm-tweets-classifier/dataset/training_dataset/harmful.csv')
test_normal = pd.read_csv('/Users/cypruscodes/Desktop/Bandar_Project/ltsm-tweets-classifier/dataset/testing_dataset/testing-normal.csv')
test_harmful = pd.read_csv('/Users/cypruscodes/Desktop/Bandar_Project/ltsm-tweets-classifier/dataset/testing_dataset/testing-harmful.csv')

# Add labels
train_normal['label'] = 0
train_harmful['label'] = 1
test_normal['label'] = 0
test_harmful['label'] = 1

# Combine datasets
train_data = pd.concat([train_normal, train_harmful], axis=0)
test_data = pd.concat([test_normal, test_harmful], axis=0)

# Shuffle datasets
train_data = train_data.sample(frac=1).reset_index(drop=True)
test_data = test_data.sample(frac=1).reset_index(drop=True)

# Ensure all tweet values are strings and drop any rows with missing values
train_data['tweet'] = train_data['tweet'].astype(str)
test_data['tweet'] = test_data['tweet'].astype(str)

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data['tweet'])

# Convert texts to sequences and pad them
train_sequences = tokenizer.texts_to_sequences(train_data['tweet'])
test_sequences = tokenizer.texts_to_sequences(test_data['tweet'])
max_length = max(len(x) for x in train_sequences)  # Max length from training data
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post')

# Build the LSTM Model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

# Manually build the model to initialize the layers properly
model.build(input_shape=(None, max_length))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Implement early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the Model
model.fit(train_padded, train_data['label'], epochs=20, validation_split=0.1, callbacks=[early_stopping])

# Evaluate the model on the testing data
predictions = model.predict(test_padded).round()

# Calculate metrics
accuracy = accuracy_score(test_data['label'], predictions)
precision = precision_score(test_data['label'], predictions)
recall = recall_score(test_data['label'], predictions)
f1 = f1_score(test_data['label'], predictions)

# Print evaluation results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save the trained model using the recommended format
model.save('/Users/cypruscodes/Desktop/Bandar_Project/ltsm-tweets-classifier/trained-models/tweet_classifier_model.keras')

# Save the tokenizer to a JSON file for later use
tokenizer_json = tokenizer.to_json()
with open('/Users/cypruscodes/Desktop/Bandar_Project/ltsm-tweets-classifier/trained-models/tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

# Save the max_length used during training
with open('/Users/cypruscodes/Desktop/Bandar_Project/ltsm-tweets-classifier/trained-models/max_length.json', 'w') as f:
    json.dump(max_length, f)
