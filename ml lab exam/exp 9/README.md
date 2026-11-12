# Experiment 9 — SVM Classification (+ Regression Residual)

## Aim
Build a complete **Support Vector Machine** classification answer: libraries, load data, EDA, plots, train/test split, confusion matrix. The last section also demonstrates a **regression residual plot**, as asked in the exam.

## Dataset
- **Exam version (this script):** Breast Cancer dataset, built into sklearn — 569 rows × 31 columns (30 features + target), no missing values.
- **Lab version:** `7,8,9/emails.csv` — 5728 emails, TF-IDF features, spam/ham target. A short lab snippet is below.

## Libraries
`pandas`, `matplotlib`, `scikit-learn` (`load_breast_cancer`, `load_diabetes`, `train_test_split`, `StandardScaler`, `SVC`, `LinearRegression`, `confusion_matrix`, `classification_report`, `ConfusionMatrixDisplay`, `mean_squared_error`, `r2_score`).

## Steps in `exp9.py`
1. **Libraries** — import everything at the top.
2. **Load** — `load_breast_cancer(as_frame=True)`, then `X = df.drop(columns="target")`, `y = df["target"]`.
3. **EDA** — `shape`, `dtypes`, `info()`, `describe()`, missing values.
4. **Plots** — bar chart of class counts, box plot of scaled features, IQR outlier count.
5. **Split** — 80/20 with `stratify=y`; scale using `StandardScaler` (fit on train only).
6. **SVM** — `SVC(kernel="rbf", C=1.0, gamma="scale")`.
7. **Evaluate** — accuracy, confusion matrix, classification report, plotted confusion matrix.
8. **Regression residual** — Linear Regression on the diabetes dataset; plot `actual − predicted` against predicted values; print MSE and R².

## Key code
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)
model = SVC(kernel="rbf", C=1.0, gamma="scale").fit(X_train, y_train)
cm = confusion_matrix(y_test, model.predict(X_test))
residuals = y_test_reg - reg.predict(X_test_reg)
```

## Expected output
- Accuracy ≈ **0.9825** (98.25%)
- Confusion matrix:
```
[[41  1]
 [ 1 71]]
```
- Regression MSE ≈ **2900.19**
- Regression R² ≈ **0.4526**

## Lab version (emails + TF-IDF)
```python
df = pd.read_csv("emails.csv")
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["text"], df["spam"], test_size=0.2, random_state=42, stratify=df["spam"])
vec = TfidfVectorizer(max_features=1000)
X_train = vec.fit_transform(X_train_text).toarray()
X_test = vec.transform(X_test_text).toarray()
grid = GridSearchCV(SVC(), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}, cv=3)
grid.fit(X_train, y_train)
print(grid.best_params_, grid.score(X_test, y_test))
```
The lab also used `PCA(n_components=2)` to plot the SVM decision boundary.

## Viva questions
1. **Why scale before SVM?** SVM maximizes margin using distances; unscaled large-range features dominate.
2. **What are support vectors?** The training points closest to the hyperplane; they define the margin.
3. **What does `C` control?** Penalty for misclassification. High `C` → fewer training errors but overfitting risk.
4. **What does `gamma` control?** Reach of a single training point for the RBF kernel. High gamma → tight, wiggly boundary.
5. **Why use the RBF kernel?** It handles non-linear boundaries by implicitly mapping data to a higher dimension.
6. **What is a residual plot?** A plot of `actual − predicted`; random scatter around 0 indicates a good fit.

## Common mistake
Fitting the scaler on the full dataset before splitting (data leakage). Also, do not report only accuracy — always include the confusion matrix and per-class metrics.
