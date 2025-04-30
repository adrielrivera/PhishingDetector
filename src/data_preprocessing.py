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
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))

    def load_datasets(self):
        # load each dataset from csv files and return as a dictionary
        datasets = {
            'phishing': pd.read_csv('phishing_email.csv'),
            'enron': pd.read_csv('Enron.csv'),
            'nigerian': pd.read_csv('Nigerian_Fraud.csv'),
            'spam': pd.read_csv('SpamAssasin.csv'),
            'ling': pd.read_csv('Ling.csv')
        }
        return datasets

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
        # load, clean, and combine all datasets, then split into train and test sets
        datasets = self.load_datasets()
        processed_data = []
        for source, df in datasets.items():
            # determine the text and label columns based on dataset structure
            text_col = 'body' if 'body' in df.columns else 'text' if 'text' in df.columns else 'content'
            label_col = 'label' if 'label' in df.columns else 'class' if 'class' in df.columns else 'is_phishing'
            for _, row in df.iterrows():
                text = str(row[text_col])
                label = int(row[label_col])
                cleaned_text = self.clean_text(text)  # clean the email text
                features = self.extract_features(cleaned_text)  # extract features from text
                features['cleaned_text'] = cleaned_text  # add cleaned text
                features['label'] = label  # add label
                features['source'] = source  # add source dataset name
                processed_data.append(features)
        final_df = pd.DataFrame(processed_data)  # create dataframe from processed data
        train_df, test_df = train_test_split(final_df, test_size=0.2, random_state=42)  # split data
        return train_df, test_df 