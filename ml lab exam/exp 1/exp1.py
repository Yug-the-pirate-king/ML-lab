# Experiment 1: EDA + Logistic Regression (Student Performance)
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

DATA = Path(__file__).resolve().parents[2] / "exp 1" / "student_performance_dataset.csv"
df = pd.read_csv(DATA)

# ---------- 1. Data analysis ----------
print("SIZE:", df.shape)
print(df.info())
print(df.describe())
print("MISSING VALUES:\n", df.isnull().sum())

# ---------- 2. Clean + encode ----------
df = df.drop(columns=["student_id", "final_exam_score"])  # score defines the grade -> leakage
num = df.select_dtypes("number").columns
cat = df.select_dtypes(exclude="number").columns.drop("final_grade")
df[num] = df[num].fillna(df[num].median())
df[cat] = df[cat].fillna(df[cat].mode().iloc[0])
df = pd.get_dummies(df, columns=cat, drop_first=True)

X, y = df.drop(columns="final_grade"), df["final_grade"]

# ---------- 3. Plots ----------
df["final_grade"].value_counts().plot.bar(title="Grade distribution")
plt.show()
df.boxplot(column="previous_grade", by="final_grade")
plt.suptitle("")
plt.title("Previous grade by final grade")
plt.show()

# ---------- 4. Split + model ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
print("5-fold CV accuracy:", cross_val_score(model, X_train, y_train, cv=5).mean())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()
