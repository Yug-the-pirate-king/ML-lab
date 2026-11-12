# Experiment 4: KNN Classification (Sonar)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

DATA = Path(__file__).resolve().parents[2] / "exp 4" / "sonar.csv"
df = pd.read_csv(DATA, header=None)          # file has no header row

# ---------- 1. Data analysis ----------
print("SIZE:", df.shape)
print(df.info())
print("MISSING:", df.isnull().sum().sum())
print(df.iloc[:, -1].value_counts())

# ---------- 2. Prepare data ----------
X = df.iloc[:, :-1]                          # first 60 columns = features
y = df.iloc[:, -1]                           # last column = M / R
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------- 3. Choose k ----------
accuracies = []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    accuracies.append(knn.score(X_test, y_test))

plt.plot(range(1, 21), accuracies, marker="o")
plt.xlabel("k (number of neighbours)")
plt.ylabel("Test accuracy")
plt.title("Choosing k for KNN")
plt.xticks(range(1, 21))
plt.show()

best_k = int(np.argmax(accuracies)) + 1
print("Best k:", best_k, "Accuracy:", max(accuracies))

# ---------- 4. Final model + confusion matrix ----------
model = KNeighborsClassifier(n_neighbors=best_k).fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=["Mine", "Rock"], cmap="Blues")
plt.title("Confusion Matrix - KNN")
plt.show()

# ---------- 5. PCA 2D view ----------
X_pca = PCA(n_components=2).fit_transform(X_train)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=(y_train == "R"),
            cmap="coolwarm", edgecolor="k")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA 2D - Sonar (red = Rock)")
plt.show()
