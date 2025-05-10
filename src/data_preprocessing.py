import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class EmailPreprocessor:
    def __init__(self):
        # download required nltk data if not already present
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            print("Downloading required NLTK data...")
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('punkt_tab')
            print("NLTK data download complete")
        
        self.stop_words = set(stopwords.words('english'))

    def load_datasets(self):
        # Only load the combined phishing_email.csv dataset
        return {'phishing': pd.read_csv('phishing_email.csv')}

    def clean_text(self, text):
        # clean and preprocess text: lowercase, remove emails, urls, special chars, and stopwords
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'\S+@\S+', ' EMAIL ', text)  # replace email addresses
        text = re.sub(r'http\S+|www\S+', ' URL ', text)  # replace urls
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # remove special characters and numbers
        text = re.sub(r'\s+', ' ', text).strip()  # remove extra whitespace
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words]  # remove stopwords
        return ' '.join(tokens)

    def extract_features(self, text):
        # extract simple statistical features from the cleaned text
        features = {}
        features['url_count'] = len(re.findall(r'URL', text))  # count url tokens
        features['email_count'] = len(re.findall(r'EMAIL', text))  # count email tokens
        features['text_length'] = len(text)  # total character length
        features['word_count'] = len(text.split())  # total word count
        words = text.split()
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0  # average word length
        return features

    def prepare_data(self):
        # load and process only the combined dataset
        datasets = self.load_datasets()
        processed_data = []
        for source, df in datasets.items():
            print(f"Processing dataset: {source}")
            print(f"Available columns: {df.columns.tolist()}")
            text_col = 'text_combined'
            label_col = 'label'
            print(f"Using columns - Text: {text_col}, Label: {label_col}")
            for _, row in df.iterrows():
                try:
                    text = str(row[text_col])
                    label = int(row[label_col])
                    cleaned_text = self.clean_text(text)
                    features = self.extract_features(cleaned_text)
                    features['cleaned_text'] = cleaned_text
                    features['label'] = label
                    features['source'] = source
                    processed_data.append(features)
                except Exception as e:
                    print(f"Error processing row in {source}: {str(e)}")
                    continue
        if not processed_data:
            raise ValueError("No data was successfully processed from the combined dataset")
        final_df = pd.DataFrame(processed_data)
        train_df, test_df = train_test_split(final_df, test_size=0.2, random_state=42)
        return train_df, test_df 