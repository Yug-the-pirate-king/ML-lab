# Experiment 4 — KNN Classification (Sonar: Mine vs Rock)

## Aim
Classify sonar returns as **Mine (M)** or **Rock (R)** using K-Nearest Neighbours, tune the number of neighbours `k`, and visualize the data in 2D with PCA.

## Dataset
`exp 4/sonar.csv` — 208 rows × 61 columns. The file has **no header row**: 60 numeric features + 1 label (`M` or `R`). Classes: M = 111, R = 97. No missing values.

## Libraries
`pandas`, `numpy`, `matplotlib`, `scikit-learn` (`train_test_split`, `StandardScaler`, `KNeighborsClassifier`, `PCA`, `ConfusionMatrixDisplay`).

## Steps
1. Load with `pd.read_csv(..., header=None)` — otherwise the first data row becomes the header.
2. `X = df.iloc[:, :-1]`, `y = df.iloc[:, -1]`.
3. Split 80/20 with `stratify=y`, then scale (KNN is distance-based).
4. Loop `k = 1..20`, fit a KNN for each, and record test accuracy.
5. Pick the best `k`, train the final model, and show the classification report + confusion matrix.
6. Reduce the 60 features to 2 with PCA and plot the classes.

## Key code
```python
df = pd.read_csv(DATA, header=None)
X, y = df.iloc[:, :-1], df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)
for k in range(1, 21):
    acc = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train).score(X_test, y_test)
```

## Expected output
- Best **k = 1**
- Test accuracy ≈ **0.8095** (81%)
- Precision/recall ≈ 0.80–0.82 for both classes
- The PCA scatter shows partial overlap between Mine and Rock.

## Viva questions
1. **Why scale features for KNN?** KNN uses distance; a large-range feature would dominate the distance.
2. **Why try odd values of k?** With two classes, an odd `k` avoids ties.
3. **Why can `k=1` overfit?** It copies the nearest training point, including noise; a larger `k` smooths the boundary.
4. **What is the curse of dimensionality?** In high dimensions, distances become similar, so nearest neighbours are less meaningful. PCA helps compress the features.
5. **How does PCA help here?** It reduces 60 features to 2 for visualization and can remove correlated noise.

## Common mistake
Loading `sonar.csv` without `header=None`, which loses one row and turns numeric column names into strings.
