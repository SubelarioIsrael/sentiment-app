import pandas as pd
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# ========== 1. LOAD DATA ==========
df = pd.read_csv("train.csv", encoding="ISO-8859-1")

# Use only the necessary columns
df = df[['text', 'sentiment']].dropna()

# ========== 2. TEXT CLEANING FUNCTION ==========
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove links
    text = re.sub(r"@\w+|#", "", text)                   # remove mentions/hashtags
    text = re.sub(r"[^a-z\s]", "", text)                 # remove non-letter chars
    text = re.sub(r"\s+", " ", text).strip()             # normalize spaces
    return text

df["clean_text"] = df["text"].apply(clean_text)

# ========== 3. SPLIT DATA ==========
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
)

# ========== 4. FEATURE EXTRACTION ==========
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),       # unigrams + bigrams
    max_features=20000,      # reasonable cap
    sublinear_tf=True,
    stop_words='english'
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ========== 5. CLASS WEIGHTING ==========
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights_dict = dict(zip(classes, class_weights))

# ========== 6. MODEL TRAINING ==========
model = LogisticRegression(
    max_iter=1000,
    class_weight=class_weights_dict,
    solver='lbfgs',
    n_jobs=-1
)
model.fit(X_train_tfidf, y_train)

# ========== 7. EVALUATION ==========
y_pred = model.predict(X_test_tfidf)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# ========== 8. SAVE MODEL + VECTORIZER ==========
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n✅ Model training complete. Files saved as:")
print(" - sentiment_model.pkl")
print(" - vectorizer.pkl")
