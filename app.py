from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

DATASET_PATH = "quora_duplicate_questions.csv"
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "tfidf.pkl"

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def train_model():
    df = pd.read_csv(DATASET_PATH).dropna()
    df["question1"] = df["question1"].apply(preprocess)
    df["question2"] = df["question2"].apply(preprocess)

    tfidf = TfidfVectorizer(max_features=5000)
    q1_tfidf = tfidf.fit_transform(df["question1"])
    q2_tfidf = tfidf.transform(df["question2"])

    import scipy.sparse as sp
    X = sp.hstack((q1_tfidf, q2_tfidf))
    y = df["is_duplicate"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(tfidf, f)

    return "Model trained successfully!"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    print("Training model...")
    train_model()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    tfidf = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/check_duplicate", methods=["POST"])
def check_duplicate():
    data = request.get_json()
    q1 = preprocess(data["question1"])
    q2 = preprocess(data["question2"])

    X = tfidf.transform([q1, q2]).toarray().reshape(1, -1)

    prediction = model.predict(X)[0]
    result = "Duplicate" if prediction == 1 else "Not Duplicate"

    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
