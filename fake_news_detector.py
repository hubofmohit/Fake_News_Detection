# fake_news_detector.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load the dataset
df = pd.read_csv("fake_or_real_news_500.csv")

# 2. Preview the data
print("Data preview:")
print(df.head())
print("\nLabel distribution:")
print(df['label'].value_counts())

# 3. Encode labels: FAKE → 0, REAL → 1
df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})

# Drop rows with NaN labels (in case any remain)
df = df.dropna(subset=['label'])

# 4. Split the dataset
X = df['text']  # You can also include 'title' if you want
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 7. Make predictions
y_pred = model.predict(X_test_vec)

# 8. Evaluate the model
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
