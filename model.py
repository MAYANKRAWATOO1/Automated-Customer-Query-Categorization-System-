import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

class QueryClassifier:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=500)
        self.label_encoder = LabelEncoder()

        self.lr = LogisticRegression(max_iter=1000)
        self.svm = SVC(probability=True)
        self.kmeans = KMeans(n_clusters=3)
        self.ann = MLPClassifier(max_iter=1000)

        self.is_trained = False

    def train(self, df):
        queries = df['Query'].apply(preprocess_text)
        X = self.vectorizer.fit_transform(queries).toarray()  # type: ignore
        y = self.label_encoder.fit_transform(df['Category'])

        self.lr.fit(X, y)
        self.svm.fit(X, y)
        self.kmeans.fit(X)
        self.ann.fit(X, y)

        self.is_trained = True

    def predict(self, query):
        q = preprocess_text(query)
        X = self.vectorizer.transform([q]).toarray()  # type: ignore

        lr = self.label_encoder.inverse_transform(self.lr.predict(X))[0]
        svm = self.label_encoder.inverse_transform(self.svm.predict(X))[0]
        ann = self.label_encoder.inverse_transform(self.ann.predict(X))[0]
        km = int(self.kmeans.predict(X)[0])

        return {
            "logistic_regression": lr,
            "svm": svm,
            "ann": ann,
            "kmeans_cluster": km,
            "final_category": svm,
            "feature_vector": X.astype("float32")
        }

    def get_all_vectors(self, queries):
        queries = [preprocess_text(q) for q in queries]
        return self.vectorizer.transform(queries).toarray().astype("float32")  # type: ignore