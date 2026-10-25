# train_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import your FeatureExtraction class
from feature import FeatureExtraction

# Function to extract features from URLs
def extract_features_from_urls(url_list, labels):
    """
    Extract features from a list of URLs
    url_list: List of URLs
    labels: List of corresponding labels (1 for safe, -1 for unsafe)
    Returns: DataFrame with features
    """
    all_features = []
    
    for i, url in enumerate(url_list):
        try:
            print(f"Processing URL {i+1}/{len(url_list)}: {url}")
            # Extract features using your FeatureExtraction class
            feature_extractor = FeatureExtraction(url)
            features = feature_extractor.getFeaturesList()
            
            # Add the label
            features.append(labels[i])
            all_features.append(features)
        except Exception as e:
            print(f"Error processing URL {url}: {e}")
            # Add default features if error
            default_features = [0] * 30  # 30 features + label
            default_features.append(labels[i])
            all_features.append(default_features)
    
    # Create column names
    column_names = [
        'UsingIP', 'LongURL', 'ShortURL', 'Symbol@', 'Redirecting//', 
        'PrefixSuffix-', 'SubDomains', 'HTTPS', 'DomainRegLen', 'Favicon',
        'NonStdPort', 'HTTPSDomainURL', 'RequestURL', 'AnchorURL', 
        'LinksInScriptTags', 'ServerFormHandler', 'InfoEmail', 'AbnormalURL',
        'WebsiteForwarding', 'StatusBarCust', 'DisableRightClick', 
        'UsingPopupWindow', 'IframeRedirection', 'AgeofDomain', 'DNSRecording',
        'WebsiteTraffic', 'PageRank', 'GoogleIndex', 'LinksPointingToPage', 
        'StatsReport', 'class'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(all_features, columns=column_names)
    return df

# Sample data for demonstration
# Replace this with your actual data
print("Creating sample data for demonstration...")

# Sample safe URLs (you can replace with your actual data)
safe_urls = [
    "https://www.google.com",
    "https://www.youtube.com",
    "https://www.facebook.com",
    "https://www.amazon.com",
    "https://www.microsoft.com"
]

# Sample unsafe URLs (you can replace with your actual data)
unsafe_urls = [
    "http://test-phishing-site.com",
    "http://fake-bank-login.com",
    "http://suspicious-site.tk",
    "http://verify-account-now.ml",
    "http://update-password.ga"
]

# Combine URLs and labels
all_urls = safe_urls + unsafe_urls
all_labels = [1] * len(safe_urls) + [-1] * len(unsafe_urls)  # 1 for safe, -1 for unsafe

# Extract features
print("Extracting features from URLs...")
data = extract_features_from_urls(all_urls, all_labels)

print(f"Data shape after feature extraction: {data.shape}")
print(f"Columns: {list(data.columns)}")
print(f"Class distribution:\n{data['class'].value_counts()}")

# If you have a CSV file with URLs and labels, you can load it instead:
try:
    # Try to load from CSV if available
    csv_data = pd.read_csv("phishing.csv")
    print(f"Loaded CSV data. Shape: {csv_data.shape}")
    data = csv_data
except:
    print("CSV file not found, using extracted features data")

# Balance the dataset by oversampling the minority class
from sklearn.utils import resample

# Separate majority and minority classes
df_majority = data[data['class'] == 1]   # Safe
df_minority = data[data['class'] == -1]  # Unsafe

print(f"Majority class (safe) samples: {len(df_majority)}")
print(f"Minority class (unsafe) samples: {len(df_minority)}")

# If we have very few unsafe samples, oversample them
if len(df_minority) < len(df_majority) / 2:
    # Upsample minority class
    df_minority_upsampled = resample(df_minority, 
                                     replace=True,     # sample with replacement
                                     n_samples=int(len(df_majority) * 0.8),  # to match 80% of majority class
                                     random_state=42) # reproducible results
    
    # Combine majority class with upsampled minority class
    data_upsampled = pd.concat([df_majority, df_minority_upsampled])
    
    # Display new class counts
    print("Class distribution after balancing:")
    print(data_upsampled['class'].value_counts())
    
    # Use the balanced dataset
    data = data_upsampled

# Split the data
print("Splitting data...")
X = data.drop('class', axis=1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train class distribution: {y_train.value_counts()}")
print(f"Test class distribution: {y_test.value_counts()}")

# Train the model with adjusted parameters
print("Training Gradient Boosting Classifier...")
gbc = GradientBoostingClassifier(
    max_depth=5,          # Increased depth
    learning_rate=0.5,     # Reduced learning rate
    n_estimators=200,      # Increased number of trees
    min_samples_split=5,    # Minimum samples required to split
    min_samples_leaf=2,     # Minimum samples required at a leaf node
    random_state=42
)
gbc.fit(X_train, y_train)

# Evaluate the model
print("Evaluating model...")
y_pred = gbc.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)

print(f"Model Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Print confusion matrix
print("Confusion Matrix:")
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)

# Print classification report
print("Classification Report:")
print(metrics.classification_report(y_test, y_pred))

# Save the model
print("Saving model...")
with open('newmodel.pkl', 'wb') as file:
    pickle.dump(gbc, file)
print("Model saved as 'newmodel.pkl'")

# Create a simple feature importance plot
print("Creating feature importance plot...")
feature_importance = gbc.feature_importances_
feature_names = X.columns

# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Top 10 Most Important Features')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance plot saved as 'feature_importance.png'")

print("Training completed successfully!")