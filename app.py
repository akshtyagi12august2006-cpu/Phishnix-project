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

    # Create a simple model for demonstration
    X_train = np.random.rand(100, 30)
    y_train = np.random.choice([1, -1], size=100)

    model = GradientBoostingClassifier(
        n_estimators=10,
        learning_rate=0.1,
        max_depth=1,
        random_state=42
    )

    model.fit(X_train, y_train)
    print("Created and trained a simple fallback model")


# Simple rule-based check for known safe/unsafe domains
# Kept for reference/testing, but NOT used to override the ML model.
def simple_rule_based_check(url):
    """
    Simple rule-based check for known safe/unsafe domains.
    Returns: 1 for safe, -1 for unsafe.
    """
    safe_domains = [
        "google.com", "youtube.com", "facebook.com", "amazon.com",
        "twitter.com", "instagram.com", "linkedin.com", "wikipedia.org",
        "yahoo.com", "reddit.com", "netflix.com", "microsoft.com",
        "github.com", "stackoverflow.com", "quora.com", "medium.com"
    ]

    unsafe_domains = [
        "test-phishing-site.com",
        "fake-bank-login.com",
        "phishing-example.com"
    ]

    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc

        for safe_domain in safe_domains:
            if safe_domain in domain:
                return 1

        for unsafe_domain in unsafe_domains:
            if unsafe_domain in domain:
                return -1

        return -1

    except:
        return -1


# Routes
@app.route("/")
def home():
    return render_template("index.html")


@app.route('/result', methods=['POST', 'GET'])
def predict():
    if request.method == "POST":
        url = request.form["name"]

        try:
            print(f"Processing URL: {url}")

            # Extract features from the URL
            obj = FeatureExtraction(url)
            features = obj.getFeaturesList()

            # Ensure exactly 30 features
            features = features[:30]

            while len(features) < 30:
                features.append(0)

            x = np.array(features).reshape(1, 30)

            print(f"Features extracted: {features[:5]}...")

            # Make prediction using the trained ML model
            y_pred = model.predict(x)[0]

            print(f"Raw model prediction: {y_pred}")

            # Get prediction probability if available
            try:
                y_proba = model.predict_proba(x)[0]
                print(f"Prediction probabilities: {y_proba}")
            except Exception:
                print("Model does not provide prediction probabilities")

            # Final prediction comes directly from the ML model
            print(f"Final prediction: {y_pred}")

            # Convert prediction into safety status
            name = convertion(url, y_pred)

            print(f"Final result: {name}")

            return render_template("index.html", name=name)

        except Exception as e:
            print(f"Error processing URL: {str(e)}")

            return render_template(
                "index.html",
                name="Error while processing URL"
            )


@app.route('/usecases', methods=['GET', 'POST'])
def usecases():
    return render_template('usecases.html')


@app.route('/test')
def test():
    # Test with a known safe URL
    test_url = "https://www.google.com"
    obj = FeatureExtraction(test_url)
    features = obj.getFeaturesList()

    features = features[:30]

    while len(features) < 30:
        features.append(0)

    x = np.array(features).reshape(1, 30)
    y_pred = model.predict(x)[0]

    # Test with a known unsafe URL
    test_url2 = "http://test-phishing-site.com"
    obj2 = FeatureExtraction(test_url2)
    features2 = obj2.getFeaturesList()

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