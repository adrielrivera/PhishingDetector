import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, Embedding, LSTM, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

class PhishingDetector:
    def __init__(self, max_words=10000, max_len=200):
        # initialize tokenizer and model parameters
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_words)
        self.model = None

    def build_model(self, feature_dim):
        # build a neural network that combines text (lstm) and numerical features
        text_input = Input(shape=(self.max_len,))
        embedding = Embedding(self.max_words, 100)(text_input)
        lstm = LSTM(64)(embedding)
        numerical_input = Input(shape=(feature_dim,))  # input for numerical features
        combined = Concatenate()([lstm, numerical_input])  # concatenate lstm and numerical features
        dense1 = Dense(64, activation='relu')(combined)
        dropout1 = Dropout(0.3)(dense1)
        dense2 = Dense(32, activation='relu')(dropout1)
        dropout2 = Dropout(0.3)(dense2)
        output = Dense(1, activation='sigmoid')(dropout2)  # output probability of phishing
        self.model = Model(inputs=[text_input, numerical_input], outputs=output)
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def prepare_features(self, df):
        # convert cleaned text to padded sequences and extract numerical features
        texts = df['cleaned_text'].values
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        text_features = pad_sequences(sequences, maxlen=self.max_len)
        numerical_features = df[['url_count', 'email_count', 'text_length', 
                               'word_count', 'avg_word_length']].values
        return text_features, numerical_features

    def train(self, train_df, val_df, epochs=10, batch_size=32):
        # train the model using training and validation data
        X_text_train, X_num_train = self.prepare_features(train_df)
        X_text_val, X_num_val = self.prepare_features(val_df)
        y_train = train_df['label'].values
        y_val = val_df['label'].values
        if self.model is None:
            self.build_model(X_num_train.shape[1])  # build model if not already built
        history = self.model.fit(
            [X_text_train, X_num_train],
            y_train,
            validation_data=([X_text_val, X_num_val], y_val),
            epochs=epochs,
            batch_size=batch_size
        )
        return history

    def predict(self, text, numerical_features):
        # predict the probability that a new email is phishing
        sequence = self.tokenizer.texts_to_sequences([text])
        text_features = pad_sequences(sequence, maxlen=self.max_len)
        prediction = self.model.predict([text_features, numerical_features.reshape(1, -1)])
        return prediction[0][0]  # return probability of being phishing

    def save_model(self, path):
        # save the trained model to disk
        self.model.save(path)

    def load_model(self, path):
        # load a previously saved model from disk
        self.model = tf.keras.models.load_model(path) 