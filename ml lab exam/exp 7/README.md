# Experiment 7 — Gaussian Naive Bayes (Manual vs scikit-learn)

## Aim
Implement **Gaussian Naive Bayes** from scratch and compare it with sklearn's `GaussianNB` on a spam/ham email dataset.

## Dataset
`7,8,9/emails.csv` — 5728 emails × 2 columns (`text`, `spam`). Classes: Ham (0) = 4360, Spam (1) = 1368. No missing values.

## Libraries
`pandas`, `numpy`, `matplotlib`, `scikit-learn` (`train_test_split`, `TfidfVectorizer`, `GaussianNB`, `accuracy_score`, `ConfusionMatrixDisplay`).

## Steps
1. Load the CSV and inspect the class counts.
2. Split the text 80/20 with `stratify=y`.
3. Fit **TF-IDF** (`max_features=500`) on the training text only, then transform the test text.
4. **Manual model:** for each class, compute the mean, variance, and prior of every feature.
5. Predict by adding the log-prior to the sum of log Gaussian likelihoods, then pick the class with the higher score.
6. Train sklearn's `GaussianNB` on the same data and compare accuracies.
7. Show the classification report and confusion matrix.

## Key formulas
- Bayes: `P(class | x) ∝ P(x | class) × P(class)`
- Gaussian likelihood: `exp(-(x − μ)² / (2σ²)) / sqrt(2πσ²)`
- Use **log** to avoid numerical underflow: `log P(x | class) + log P(class)`

## Key code
```python
vec = TfidfVectorizer(max_features=500)
X_train = vec.fit_transform(X_train_text).toarray()
X_test = vec.transform(X_test_text).toarray()
for c in np.unique(y_train):
    Xc = X[y == c]
    mean, var, prior = Xc.mean(0), Xc.var(0) + 1e-9, len(Xc) / len(y)
log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var) + (X - mean) ** 2 / var, axis=1)
```

## Expected output
- Manual accuracy: **0.9686**
- Sklearn accuracy: **0.9686**
- Ham: precision 0.99, recall 0.97
- Spam: precision 0.92, recall 0.96

The two implementations agree closely, which confirms the manual formula.

## Viva questions
1. **Why is it called “naive”?** It assumes all features are independent given the class, which is rarely true for words.
2. **Why Gaussian?** It assumes each feature follows a normal distribution within each class. For text counts, Multinomial NB is often more natural.
3. **Why add `1e-9` to the variance?** To avoid division by zero for features that are constant in a class.
4. **Why use logs?** Multiplying many small probabilities underflows to 0; adding logs is numerically stable.
5. **Why TF-IDF instead of raw counts?** It down-weights common words and highlights informative words.

## Common mistake
Forgetting `return self` at the end of a custom `fit` method, so the model object becomes `None`.
