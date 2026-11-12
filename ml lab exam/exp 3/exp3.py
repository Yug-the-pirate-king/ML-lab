# Experiment 3: Decision Tree Classification (Calorie Efficiency)
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

DATA = Path(__file__).resolve().parents[2] / "exp 3" / "calorie_efficiency_dataset.csv"
df = pd.read_csv(DATA, nrows=30000)   # full file has 1M rows; sample for speed

# ---------- 1. Data analysis ----------
print("SIZE:", df.shape)
print(df.info())
print(df.describe())
print("MISSING:", df.isnull().sum().sum())
print(df["calorie_efficiency"].value_counts())

# ---------- 2. Prepare data ----------
le = LabelEncoder()
X = df.drop(columns="calorie_efficiency")
y = le.fit_transform(df["calorie_efficiency"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# ---------- 3. Compare unpruned / depth-limited / pruned tree ----------
models = {
    "Unpruned": DecisionTreeClassifier(random_state=42),
    "Depth=5": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Pruned": DecisionTreeClassifier(ccp_alpha=0.01, random_state=42),
}
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name}: train={model.score(X_train, y_train):.4f}, "
          f"test={model.score(X_test, y_test):.4f}")

# ---------- 4. Evaluate best simple model ----------
model = models["Depth=5"]
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=le.classes_, cmap="Blues")
plt.xticks(rotation=45)
plt.title("Confusion Matrix - Decision Tree")
plt.show()

plot_tree(model, max_depth=3, feature_names=X.columns, class_names=le.classes_,
          filled=True, rounded=True, fontsize=7)
plt.title("Decision Tree (top 3 levels)")
plt.show()
