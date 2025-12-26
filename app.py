from flask import Flask, render_template, request, redirect, url_for, session, flash
import pickle
import numpy as np
import os

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')

# Load the model and vectorizer
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    print("Model or vectorizer not found. Please run train_model.py first.")
    model = None
    vectorizer = None

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

@app.route('/', methods=['GET', 'POST'])
def index():
    # Require login
    if 'user' not in session:
        return redirect(url_for('login'))

    prediction = None
    email_text = ""
    
    if request.method == 'POST':
        email_text = request.form['email']
        
        # 1. Try Heuristic First (Hybrid Approach)
        heuristic_pred = auto_label(email_text)
        
        if heuristic_pred != 'General':
            prediction = heuristic_pred
            print(f"Heuristic Prediction: {prediction}")
        elif model and vectorizer:
            # 2. Fallback to AI Model
            text_vectorized = vectorizer.transform([email_text])
            prediction = model.predict(text_vectorized)[0]
            print(f"Model Prediction: {prediction}")
        else:
            prediction = "Error: Model not loaded."
            print(prediction)
            
    return render_template('index.html', prediction=prediction, email_text=email_text, model=model, username=session.get('user'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    # Simple demo login — replace with real auth in production
    demo_users = {'admin': 'password123'}
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if username in demo_users and demo_users[username] == password:
            session['user'] = username
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
            return render_template('login.html')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
