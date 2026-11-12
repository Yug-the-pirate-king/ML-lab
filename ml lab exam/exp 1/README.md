# Experiment 1 — EDA + Logistic Regression (Student Performance)

## Aim
Analyze the student performance dataset, handle missing values, encode categorical features, and predict `final_grade` (A–F) using logistic regression. Evaluate with K-Fold cross-validation and a confusion matrix.

## Dataset
`exp 1/student_performance_dataset.csv` — 1000 rows × 12 columns. Only `parental_education` has missing values (102).

## Libraries
`pandas`, `matplotlib`, `scikit-learn` (`train_test_split`, `StandardScaler`, `LogisticRegression`, `cross_val_score`, `ConfusionMatrixDisplay`).

## Steps
1. Load the CSV and print `shape`, `info()`, `describe()`, `isnull().sum()`.
2. Drop `student_id` and `final_exam_score`. The exam score directly determines the grade, so keeping it is **data leakage**.
3. Fill numeric missing values with the median and categorical missing values with the mode.
4. One-hot encode categorical columns with `pd.get_dummies(..., drop_first=True)`.
5. Split 80/20 with `stratify=y`, then scale using `StandardScaler` (fit on train only).
6. Run 5-fold cross-validation, fit the final model, and evaluate with a classification report and confusion matrix.

## Key code
```python
df = pd.get_dummies(df, columns=cat, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))
```

## Expected output
- 5-fold CV accuracy ≈ **0.57**
- Test accuracy ≈ **0.64**
- Class `F` has only 2 test samples, so its recall can be 0. This is normal for a tiny class.

## Viva questions
1. **Why impute missing values?** Most sklearn models cannot handle `NaN`; median is robust to outliers, mode works for categories.
2. **Why one-hot encode?** Models need numbers, and labels like A/B/C have no mathematical order.
3. **Why `stratify=y`?** Keeps the same grade proportion in train and test.
4. **Why drop `final_exam_score`?** It is used to calculate `final_grade`, so it leaks the answer.
5. **What does K-Fold do?** Trains on K−1 folds and validates on the remaining fold, giving a more reliable accuracy estimate.

## Common mistake
Fitting `StandardScaler` on the full dataset before splitting — that leaks test information. Fit on `X_train`, then `transform` `X_test`.
