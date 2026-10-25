# 🛡️ Phishnix - AI-Based Phishing Detection System

Phishnix is a machine learning–powered phishing detection system designed to identify malicious URLs and protect users from online scams. This project leverages supervised learning and real-world datasets to provide accurate, real-time security analysis.

---

## 🌟 Key Features

* **Intelligent Detection:** Uses supervised ML algorithms to classify URLs as safe or malicious.
* **Feature Extraction:** Analyzes lexical and structural features (URL length, symbols, protocols).
* **Web Interface:** A clean, responsive frontend built with Flask.
* **Instant Results:** Optimized model loading via pickle for lightning-fast predictions.
* **Robust Security:** Built-in input validation to handle diverse URL formats safely.

---

## 📋 Prerequisites

Before running the project, ensure you have the following installed:

* **Python:** 3.8 or higher
* **Libraries:**

  * Flask (Web Framework)
  * scikit-learn (Machine Learning)
  * pandas & numpy (Data Processing)
  * pickle (Model Serialization)

---

## 🚀 Installation & Setup

Follow these steps to get the project running on your local machine:

1. **Clone the Repository**

```bash
git clone https://github.com/iAryanbajaj/Phishnix1.2.git
cd Phishnix1.2
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the Application**

```bash
python app.py
```

4. **Access the App**
   Open your browser and navigate to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🧠 How It Works

1. **Input:** User enters a URL into the web interface.
2. **Processing:** The `feature.py` script extracts key indicators (e.g., presence of '@', URL depth, HTTPS usage).
3. **Prediction:** The pre-trained model (`newmodel.pkl`) processes these features.
4. **Output:** The system displays whether the site **Legitimate** or a **Phishing threat**.

```python
# Core prediction logic
import pickle

model = pickle.load(open('newmodel.pkl', 'rb'))
prediction = model.predict([url_features])
```

---

## 📁 Project Structure

```
Phishnix/
├── app.py                # Main Flask application
├── train_model.py        # Script for ML model training
├── feature.py            # Feature extraction logic
├── convert.py            # Data conversion utilities
├── newmodel.pkl          # Saved Machine Learning model
├── requirements.txt      # Project dependencies
├── templates/            # HTML files (index, usecases, etc.)
├── static/               # CSS, JS, and Images
└── DataFiles/            # Datasets (phishing.csv, phishurls.csv, etc.)
```

---

## 👥 Contributors

* **Aryan Bajaj** – [iAryanbajaj](https://github.com/iAryanbajaj)
* **Yashna Chugh** – [Yashna Chugh](https://github.com/YASHNACHUGH2408)

---

## 📌 Future Enhancements

* [ ] **Browser Extension:** Real-time scanning while surfing the web.
* [ ] **Deep Learning:** Implementation of Neural Networks for higher accuracy.
* [ ] **API Access:** Public API for third-party security integrations.
* [ ] **Live Database:** Auto-updating blacklist from global security feeds..

---

## 📄 License
This project is licensed under the **MIT License** – see the LICENSE file for details.
