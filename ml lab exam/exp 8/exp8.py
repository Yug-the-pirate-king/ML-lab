# Experiment 8: Logistic Regression for Binary Classification (Spam)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_curve, roc_auc_score,
                             ConfusionMatrixDisplay)

DATA = Path(__file__).resolve().parents[2] / "7,8,9" / "emails.csv"
df = pd.read_csv(DATA)

# ---------- 1. Data analysis ----------
print("SIZE:", df.shape)
print(df["spam"].value_counts(normalize=True))

# ---------- 2. Split text, then TF-IDF (fit on train only) ----------
y = df["spam"]
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["text"], y, test_size=0.2, random_state=42, stratify=y)
vectorizer = TfidfVectorizer(max_features=3000)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# ---------- 3. Train + evaluate ----------
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
print("AUC:", roc_auc_score(y_test, y_prob))

# ---------- 4. Sigmoid + ROC curve ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

z = np.linspace(-10, 10, 200)
axes[0].plot(z, 1 / (1 + np.exp(-z)))
axes[0].axhline(0.5, color="red", linestyle="--")
axes[0].axvline(0, color="black", linestyle=":")
axes[0].set_title("Sigmoid / decision boundary")
axes[0].set_xlabel("z = wx + b")
axes[0].set_ylabel("P(spam)")

fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.3f}")
axes[1].plot([0, 1], [0, 1], "k--")
axes[1].set_title("ROC curve")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend()
plt.tight_layout()
plt.show()

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=["Ham", "Spam"], cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()
