# ML Lab Exam — All 10 Experiments

Study pack for the ML lab exam. One folder per experiment, and each folder contains only:

- `README.md` — aim, steps, expected output, viva questions
- `expN.py` — minimal runnable code

## Folder map

| Exp | Topic | Dataset | Main algorithm |
|---|---|---|---|
| 1 | EDA + classification | `student_performance_dataset.csv` | Logistic Regression, K-Fold, confusion matrix |
| 2 | Regression | `employee_salary_regression.csv` | Simple + multiple Linear Regression, residuals |
| 3 | Classification + pruning | `calorie_efficiency_dataset.csv` | Decision Tree, overfitting, pruning |
| 4 | Classification | `sonar.csv` | KNN, tuning k, PCA |
| 5 | Dimensionality reduction | `Bachelors ... USA States.csv` | PCA, t-SNE, Random Forest |
| 6 | Recommender system | `Bachelors ... USA States.csv` | Cosine / Pearson similarity |
| 7 | Classification | `emails.csv` | Gaussian Naive Bayes (manual + sklearn) |
| 8 | Classification | `emails.csv` | Logistic Regression, TF-IDF, ROC/AUC |
| 9 | Classification + regression | Breast Cancer (built-in) / `emails.csv` | SVM, confusion matrix, residual |
| 10 | Neural networks | `make_blobs`, `make_moons` | Perceptron, MLP |

## How to run

Keep the `ml lab exam` folder inside `ML lab` so the dataset paths work.

```bash
cd "ml lab exam/exp 1"
python exp1.py
```

Install the libraries once:

```bash
pip install pandas numpy matplotlib scikit-learn
```

Each script opens plots one at a time — close one to see the next.

## Exam answer template (classification)

1. **Libraries** — pandas, matplotlib, sklearn.
2. **Load data** — `pd.read_csv(...)` or a built-in dataset.
3. **Data analysis** — `shape`, `dtypes`, `info()`, `describe()`, `isnull().sum()`.
4. **Visualization** — bar chart (class count), box plot (outliers), IQR rule.
5. **Preprocess** — impute missing values, encode categories, scale features.
6. **Split** — `train_test_split(..., test_size=0.2, random_state=42, stratify=y)`.
7. **Train** — fit the classifier on the training set only.
8. **Evaluate** — confusion matrix + classification report (accuracy, precision, recall, F1).

## Exam answer template (regression)

1. Load and describe the data.
2. Choose `X` (features) and `y` (continuous target).
3. Split into train/test.
4. Fit `LinearRegression`.
5. Predict and compute **RMSE** and **R²**.
6. Plot **residual = actual − predicted** against predicted values. A random cloud around 0 means the model fits well.

## Marks-saving lines

```python
df.shape, df.dtypes, df.info(), df.describe(), df.isnull().sum()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
residuals = y_test - y_pred
```

## Common mistakes to avoid

- Scaling before splitting → data leakage. Fit the scaler on `X_train` only.
- Using accuracy alone on imbalanced data → also show precision, recall, F1.
- Forgetting `stratify=y` → class ratio may change in the test set.
- Plotting residual as `predicted − actual` → standard is `actual − predicted`.
- Using `header=None` for `sonar.csv` (it has no header row).
- Forgetting to encode text/categorical columns before fitting a model.
- Fitting TF-IDF on all text before splitting → fit it on the training text only.
