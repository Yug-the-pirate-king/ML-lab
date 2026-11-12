# Experiment 8 — Logistic Regression for Binary Classification (Spam)

## Aim
Classify emails as **Ham (0)** or **Spam (1)** using logistic regression on TF-IDF features, and evaluate with a confusion matrix, ROC curve, and AUC.

## Dataset
`7,8,9/emails.csv` — 5728 emails × 2 columns (`text`, `spam`). Ham = 76.1%, Spam = 23.9%. No missing values.

## Libraries
`pandas`, `numpy`, `matplotlib`, `scikit-learn` (`train_test_split`, `TfidfVectorizer`, `LogisticRegression`, `roc_curve`, `roc_auc_score`, `ConfusionMatrixDisplay`).

## Steps
1. Load the CSV and check the class proportions.
2. Split the text 80/20 with `stratify=y`.
3. Fit TF-IDF (`max_features=3000`) on the training text only, then transform the test text.
4. Fit `LogisticRegression(max_iter=1000)`.
5. Predict class labels and probabilities.
6. Print the classification report and AUC.
7. Plot the **sigmoid** with the 0.5 decision boundary, the **ROC curve**, and the **confusion matrix**.

## Key concepts
- **Sigmoid:** `p = 1 / (1 + e^(−z))`, where `z = wx + b`. It converts a score into a probability between 0 and 1.
- **Decision boundary:** predict class 1 if `p ≥ 0.5` (equivalently `z ≥ 0`).
- **Log loss:** penalizes confident wrong probabilities.
- **ROC/AUC:** shows the trade-off between true-positive and false-positive rates; AUC 1.0 is perfect, 0.5 is random.
- **Regularization `C`:** smaller `C` = stronger regularization (simpler model).

## Key code
```python
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["text"], df["spam"], test_size=0.2, random_state=42, stratify=df["spam"])
vec = TfidfVectorizer(max_features=3000)
X_train = vec.fit_transform(X_train_text)
X_test = vec.transform(X_test_text)
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]
print("AUC:", roc_auc_score(y_test, y_prob))
```

## Expected output
- Accuracy ≈ **0.98**
- AUC ≈ **0.9987**
- Ham: precision 0.98, recall 1.00
- Spam: precision 0.98, recall 0.94

## Viva questions
1. **Why use logistic regression for spam?** It outputs calibrated probabilities and works well with high-dimensional text features.
2. **What is the sigmoid doing?** Mapping any real-valued score to a probability in (0, 1).
3. **What does AUC = 0.9987 mean?** The model ranks a random spam email above a random ham email 99.87% of the time.
4. **Why not use accuracy alone?** With 76% ham, predicting “ham” always gives 76% accuracy but misses all spam. Check precision/recall/F1.
5. **What does `C` control?** The inverse of regularization strength; high `C` fits the training data more closely.

## Common mistake
Using `predict_proba(X_test)[:, 0]` (probability of Ham) for the ROC curve. Spam is class 1, so use column `1`.
