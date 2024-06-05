import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download stopwords if not already done
nltk.download('stopwords')
nltk.download('punkt')

# Define stop words
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Remove stop words from the text."""
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Load the dataset
input_file = '/Users/cypruscodes/Desktop/Bandar_Project/ltsm-tweets-classifier/dataset/training_dataset/harmful.csv' 
output_file = '/Users/cypruscodes/Desktop/Bandar_Project/ltsm-tweets-classifier/dataset/training_dataset/harmful.csv'  

# Read the dataset into a pandas DataFrame
df = pd.read_csv(input_file)

# Apply preprocessing to the tweet column
df['tweet'] = df['tweet'].apply(preprocess_text)

# Save the processed DataFrame to a new CSV file
df.to_csv(output_file, index=False)

print(f"Processed dataset saved to {output_file}")
