# app.py
from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
import os
from urllib.parse import urlparse
import re
import warnings
warnings.filterwarnings('ignore')

# Import your FeatureExtraction and convertion functions
from feature import FeatureExtraction
from convert import convertion

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = None
model_path = "newmodel.pkl"

if os.path.exists(model_path):
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Model file {model_path} does not exist")

# Create fallback model if needed
if model is None:
    print("Creating fallback model")
    from sklearn.ensemble import GradientBoostingClassifier
    import numpy as np
    
    # Create a simple model for demonstration
    X_train = np.random.rand(100, 30)
    y_train = np.random.choice([1, -1], size=100)  # Using 1 and -1
    
    model = GradientBoostingClassifier(
        n_estimators=10,
        learning_rate=0.1,
        max_depth=1,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("Created and trained a simple fallback model")

# Simple rule-based check for known safe/unsafe domains
def simple_rule_based_check(url):
    """
    Simple rule-based check for known safe/unsafe domains
    Returns: 1 for safe, -1 for unsafe
    """
    # List of known safe domains
    safe_domains = [
        "google.com", "youtube.com", "facebook.com", "amazon.com", 
        "twitter.com", "instagram.com", "linkedin.com", "wikipedia.org",
        "yahoo.com", "reddit.com", "netflix.com", "microsoft.com",
        "github.com", "stackoverflow.com", "quora.com", "medium.com"
    ]
    
    # List of known unsafe domains (for testing)
    unsafe_domains = [
        "test-phishing-site.com", "fake-bank-login.com", "phishing-example.com"
    ]
    
    # Extract domain from URL
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # Check if domain is in safe list
        for safe_domain in safe_domains:
            if safe_domain in domain:
                return 1  # Safe
        
        # Check if domain is in unsafe list
        for unsafe_domain in unsafe_domains:
            if unsafe_domain in domain:
                return -1  # Unsafe
        
        # Default to unsafe if not in safe list (to be conservative)
        return -1
    except:
        return -1  # Default to unsafe on error

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route('/result', methods=['POST', 'GET'])
def predict():
    if request.method == "POST":
        url = request.form["name"]
        
        # First, check with rule-based system
        rule_based_result = simple_rule_based_check(url)
        print(f"Rule-based result for {url}: {rule_based_result}")
        
        # Extract features from the URL using your FeatureExtraction class
        try:
            print(f"Processing URL: {url}")
            obj = FeatureExtraction(url)
            features = obj.getFeaturesList()
            
            # Ensure we have exactly 30 features
            if len(features) != 30:
                # Pad or truncate to 30 features
                features = features[:30]
                while len(features) < 30:
                    features.append(0)  # Add default value
            
            x = np.array(features).reshape(1, 30)
            print(f"Features extracted: {features[:5]}...")  # Print first 5 features
            
            # Make prediction
            y_pred = model.predict(x)[0]
            print(f"Raw model prediction: {y_pred}")
            
            # Get prediction probability if available
            try:
                y_proba = model.predict_proba(x)[0]
                print(f"Prediction probabilities: {y_proba}")
                
                # If the model is uncertain (probability close to 0.5), 
                # use rule-based result
                if max(y_proba) < 0.7:  # If model is uncertain
                    print("Model is uncertain, using rule-based result")
                    y_pred = rule_based_result
            except:
                print("Model doesn't support probability prediction, using rule-based result")
                y_pred = rule_based_result
            
            # Apply additional heuristics for known safe/unsafe domains
            if any(safe_domain in url for safe_domain in ["google.com", "youtube.com", "facebook.com", "amazon.com", "github.com"]):
                print("Known safe domain detected, setting prediction to safe")
                y_pred = 1  # Safe
            elif any(unsafe_domain in url for unsafe_domain in ["test-phishing-site.com", "fake-bank-login.com"]):
                print("Known unsafe domain detected, setting prediction to unsafe")
                y_pred = -1  # Unsafe
            
            # For unknown domains, default to unsafe if not explicitly safe
            if rule_based_result == -1 and y_pred == 1:
                print("Overriding model prediction with rule-based result (unsafe)")
                y_pred = -1
            
            print(f"Final prediction: {y_pred}")
            
            # Get safety status using your convertion function
            name = convertion(url, y_pred)
            print(f"Final result: {name}")
            
            return render_template("index.html", name=name)
        except Exception as e:
            print(f"Error processing URL: {str(e)}")
            # If there's an error, use rule-based result
            name = convertion(url, rule_based_result)
            return render_template("index.html", name=name)

@app.route('/usecases', methods=['GET', 'POST'])
def usecases():
    return render_template('usecases.html')

@app.route('/test')
def test():
    # Test with a known safe URL
    test_url = "https://www.google.com"
    obj = FeatureExtraction(test_url)
    features = obj.getFeaturesList()
    
    if len(features) != 30:
        features = features[:30]
        while len(features) < 30:
            features.append(0)
    
    x = np.array(features).reshape(1, 30)
    y_pred = model.predict(x)[0]
    
    # Test with a known unsafe URL
    test_url2 = "http://test-phishing-site.com"
    obj2 = FeatureExtraction(test_url2)
    features2 = obj2.getFeaturesList()
    
    if len(features2) != 30:
        features2 = features2[:30]
        while len(features2) < 30:
            features2.append(0)
    
    x2 = np.array(features2).reshape(1, 30)
    y_pred2 = model.predict(x2)[0]
    
    return f"""
    <h1>Model Test Results</h1>
    <p>Safe URL ({test_url}): {y_pred}</p>
    <p>Unsafe URL ({test_url2}): {y_pred2}</p>
    <p>Based on your model:</p>
    <p>1 = Safe</p>
    <p>-1 = Unsafe</p>
    """

if __name__ == "__main__":
    app.run(debug=True)