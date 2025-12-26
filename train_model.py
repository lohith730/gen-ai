import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import os
import re

# Ensure NLTK data is downloaded
nltk.download('stopwords')

def parse_raw_message(raw_message):
    """Extracts the body from the raw email message."""
    lines = raw_message.split('\n')
    body = []
    reading_body = False
    for line in lines:
        if reading_body:
            body.append(line)
        elif line.strip() == "":
            reading_body = True
    return "\n".join(body).strip()

def auto_label(text):
    """Heuristic to label emails based on keywords."""
    text = text.lower()
    
    financial_keywords = ['budget', 'invoice', 'purchase', 'financial', 'report', 'quarterly', 'bank', 'money', 'expense', 'cost']
    urgent_keywords = ['urgent', 'immediate', 'emergency', 'deadline', 'breach', 'asap', 'critical', 'alert', 'warning']
    hr_keywords = ['hr', 'policies', 'performance', 'review', 'insurance', 'promotion', 'holiday', 'leave', 'benefits', 'hiring', 'salary']
    
    for word in urgent_keywords:
        if word in text:
            return 'Urgent'
    for word in financial_keywords:
        if word in text:
            return 'Financial'
    for word in hr_keywords:
        if word in text:
            return 'HR'
            
    return 'General'

def train():
    print("Task 1: Loading and Parsing Dataset")
    
    csv_path = 'emails.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
    try:
        df = pd.read_csv(csv_path, nrows=5000) # Read first 5000 rows for training to ensure speed
        print(f"Loaded {len(df)} emails.")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Parse the raw message to get the content
    print("Parsing email content...")
    df['parsed_content'] = df['message'].apply(parse_raw_message)
    
    # Auto-label the data
    print("Auto-labeling data...")
    df['category'] = df['parsed_content'].apply(auto_label)
    
    print("Label distribution:")
    print(df['category'].value_counts())

    print("\nTask 2 & 3: Text Preprocessing and Feature Extraction")
    # Preprocess text data
    stop_words = stopwords.words('english')
    
    # Initialize TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, lowercase=True, max_features=5000)
    
    # Fit and transform the text data
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['parsed_content'])
    print("Feature extraction completed.")

    print("\nTasks 4 & 5: Model Training and Model Evaluation")
    # Split the data
    # Stratify split to preserve label proportions in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        tfidf_matrix, df['category'], test_size=0.2, random_state=42, stratify=df['category']
    )
    
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    print("Model training completed.")

    # Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    
    print(f"\nModel Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Save the model and vectorizer
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
        
    print("\nModel and vectorizer saved to 'model.pkl' and 'vectorizer.pkl'")

if __name__ == "__main__":
    train()
