# Experiment 7: Gaussian Naive Bayes (Manual vs scikit-learn)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

DATA = Path(__file__).resolve().parents[2] / "7,8,9" / "emails.csv"
df = pd.read_csv(DATA)

# ---------- 1. Data analysis ----------
print("SIZE:", df.shape)
print(df.info())
print(df["spam"].value_counts())

# ---------- 2. Split text, then TF-IDF (fit on train only) ----------
y = df["spam"].values
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["text"], y, test_size=0.2, random_state=42, stratify=y)
vectorizer = TfidfVectorizer(max_features=500)
X_train = vectorizer.fit_transform(X_train_text).toarray()
X_test = vectorizer.transform(X_test_text).toarray()

# ---------- 3. Manual Gaussian Naive Bayes ----------
class ManualGaussianNB:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.params = []
        for c in self.classes:
            Xc = X[y == c]
            self.params.append((Xc.mean(0), Xc.var(0) + 1e-9, len(Xc) / len(y)))
        return self

    def predict(self, X):
        scores = []
        for mean, var, prior in self.params:
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * var) + (X - mean) ** 2 / var, axis=1)
            scores.append(log_likelihood + np.log(prior))
        return self.classes[np.argmax(scores, axis=0)]

manual = ManualGaussianNB().fit(X_train, y_train)
sklearn_model = GaussianNB().fit(X_train, y_train)

for name, model in [("Manual", manual), ("Sklearn", sklearn_model)]:
    print(f"{name} accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")

# ---------- 4. Evaluate + confusion matrix ----------
y_pred = sklearn_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=["Ham", "Spam"], cmap="Blues")
plt.title("Confusion Matrix - Gaussian NB")
plt.show()
