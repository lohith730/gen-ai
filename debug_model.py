import pandas as pd
import pickle
import os
from train_model import auto_label, parse_raw_message

def test_model():
    print("--- Testing Model ---")
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        print("Model and vectorizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    test_cases = [
        "This is urgent, please reply asap.",
        "Here is the invoice for the last month.",
        "I need to discuss my salary and benefits.",
        "Hey, how are you doing today?",
        "URGENT: System failure",
        "Budget report attached"
    ]

    for text in test_cases:
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        probs = model.predict_proba(vec)[0]
        print(f"Input: '{text}' -> Prediction: {pred} (Probs: {probs})")

def check_data_distribution():
    print("\n--- Checking Data Distribution ---")
    if not os.path.exists('emails.csv'):
        print("emails.csv not found.")
        return

    try:
        df = pd.read_csv('emails.csv', nrows=5000)
        df['parsed_content'] = df['message'].apply(parse_raw_message)
        df['category'] = df['parsed_content'].apply(auto_label)
        print(df['category'].value_counts())
    except Exception as e:
        print(f"Error processing data: {e}")

if __name__ == "__main__":
    test_model()
    check_data_distribution()
