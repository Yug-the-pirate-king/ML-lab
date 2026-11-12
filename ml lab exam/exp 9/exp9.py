# ============================================================
# Experiment 9: SVM Classification (+ Regression Residual)
# Dataset: Breast Cancer (built into sklearn, works offline)
# Note: the lab used emails.csv + TF-IDF; same SVM steps apply.
# Run: python exp9.py
# ============================================================

# ---------- a) LIBRARIES (2 marks) ----------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (confusion_matrix, classification_report,
                             ConfusionMatrixDisplay, mean_squared_error, r2_score)

# ---------- b) LOAD DATASET (2 marks) ----------
data = load_breast_cancer(as_frame=True)
df = data.frame                    # 569 rows x 31 cols (30 features + target)
X, y = df.drop(columns="target"), df["target"]

# ---------- c) DATA ANALYSIS (5 marks) ----------
print("SIZE      :", df.shape)              # (rows, columns)
print("DATA TYPES:\n", df.dtypes.value_counts())
print("MISSING   :", df.isnull().sum().sum(), "missing values")
print("\nINFO:")
df.info()
print("\nDESCRIBE:\n", df.describe().T)     # .T = easier to read

# ---------- c) VISUALIZATION + OUTLIER DETECTION (5 marks) ----------
# Bar chart: class balance
df["target"].value_counts().plot.bar(color=["tomato", "steelblue"],
                                     title="Class count (0=malignant, 1=benign)")
plt.show()

# Box plot: outliers (features scaled so all fit on one scale)
scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
scaled.plot.box(figsize=(14, 4), rot=90, title="Box plot of scaled features")
plt.show()

# Outlier count by IQR rule: value < Q1-1.5*IQR or > Q3+1.5*IQR
q1, q3 = X.quantile(.25), X.quantile(.75)
iqr = q3 - q1
outliers = ((X < q1 - 1.5 * iqr) | (X > q3 + 1.5 * iqr)).sum().sum()
print("Outlier values (IQR rule):", outliers)

# ---------- SPLIT DATA (2 marks) ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()                  # SVM needs scaled features
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)          # fit on train only!

# ---------- SVM MODEL ----------
model = SVC(kernel="rbf", C=1.0, gamma="scale")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", model.score(X_test, y_test))

# ---------- CONFUSION MATRIX (4 marks, compulsory) ----------
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)
print(classification_report(y_test, y_pred, target_names=data.target_names))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=data.target_names, cmap="Blues")
plt.title("Confusion Matrix - SVM")
plt.show()

# ---------- REGRESSION + RESIDUAL PLOT ----------
# (Residual = actual - predicted; random scatter around 0 = good fit)
diabetes = load_diabetes(as_frame=True)
Xr, yr = diabetes.data, diabetes.target
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=0.2, random_state=42)

reg = LinearRegression().fit(Xr_train, yr_train)
pred = reg.predict(Xr_test)
residuals = yr_test - pred

plt.scatter(pred, residuals, alpha=.7)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted value"); plt.ylabel("Residual")
plt.title("Residual Plot - Linear Regression")
plt.show()

print("Regression MSE:", mean_squared_error(yr_test, pred))
print("Regression R2 :", r2_score(yr_test, pred))
