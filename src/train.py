from data_preprocessing import EmailPreprocessor
from model import PhishingDetector
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def plot_training_history(history):
    # plot training and validation accuracy and loss for each epoch
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.title('model accuracy over epochs')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.title('model loss over epochs')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    # plot a confusion matrix for the model's predictions
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('confusion matrix')
    plt.ylabel('true label')
    plt.xlabel('predicted label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    # load and preprocess all email datasets
    print("Loading and preprocessing data...")
    preprocessor = EmailPreprocessor()
    train_df, test_df = preprocessor.prepare_data()
    # initialize the phishing detector model
    print("Initializing model...")
    detector = PhishingDetector()
    # train the model and validate on the test set
    print("Training model...")
    history = detector.train(train_df, test_df, epochs=10)
    # plot training history for accuracy and loss
    plot_training_history(history)
    # evaluate the model on the test set
    print("\nEvaluating model...")
    X_text_test, X_num_test = detector.prepare_features(test_df)
    y_test = test_df['label'].values
    predictions = detector.model.predict([X_text_test, X_num_test])
    y_pred = (predictions > 0.5).astype(int)
    # print a detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    # plot and save the confusion matrix
    plot_confusion_matrix(y_test, y_pred)
    # save the trained model to disk
    print("\nSaving model...")
    detector.save_model('phishing_detector_model.keras')
    print("\nTraining complete! Model saved as 'phishing_detector_model.keras'")

if __name__ == "__main__":
    main() 