# Experiment 5 — PCA + t-SNE + Random Forest (Education Data)

## Aim
Visualize high-dimensional education data with **PCA** and **t-SNE**, then compare a Random Forest classifier on the original features versus PCA-reduced features.

## Dataset
`exp5/Bachelors or Higher Degree for USA States.csv` — 16 years × 52 columns (date + 50 states + District of Columbia), no missing values. Each value is the percentage of adults with a bachelor's degree or higher.

The script builds a binary target: whether **New Hampshire's** value is above or below its median (8 years each). The other 50 jurisdictions are the features.

## Libraries
`pandas`, `matplotlib`, `scikit-learn` (`train_test_split`, `StandardScaler`, `RandomForestClassifier`, `PCA`, `TSNE`).

## Steps
1. Load the CSV and strip spaces from column names (many start with a space).
2. Select numeric columns; build `y` from New Hampshire's median split; `X` = other states.
3. Split 70/30 with `stratify=y` and scale the features.
4. Train a baseline Random Forest on all features.
5. Apply **PCA (2 components)** and **t-SNE (2 components)** and plot both, colored by class.
6. Apply PCA keeping **90% of variance**, retrain Random Forest, and compare accuracy and the number of components.

## Key code
```python
y = (features["New Hampshire"] >= features["New Hampshire"].median()).astype(int)
X = features.drop(columns="New Hampshire")
pca = PCA(n_components=2); X_pca = pca.fit_transform(X_train_s)
tsne = TSNE(n_components=2, perplexity=5, random_state=42).fit_transform(X_train_s)
pca90 = PCA(n_components=0.90); X_train_p = pca90.fit_transform(X_train_s)
print("Components:", pca90.n_components_)
```

## Expected output
- Baseline accuracy: **1.0**
- PCA components for 90% variance: **1**
- PCA accuracy: **1.0**

The perfect score is **not impressive**: there are only 16 rows, the test set has 5 rows, and state education values are highly correlated across years. Use this experiment to explain PCA/t-SNE, not to claim a great model.

## Viva questions
1. **PCA vs t-SNE?** PCA is linear and preserves global variance; t-SNE is non-linear and preserves local neighbourhoods, mainly for visualization.
2. **What does “PCA with 90% variance” mean?** Keep the smallest number of components that together explain 90% of the total variance.
3. **Why scale before PCA?** PCA maximizes variance; unscaled features with larger ranges would dominate.
4. **Why is accuracy 1.0 suspicious?** Tiny dataset, highly correlated features, and a 5-row test set — the estimate is unstable.
5. **Can you use t-SNE features for classification?** It is possible but not recommended; t-SNE is mainly for visualization and does not preserve distances globally.

## Common mistake
Using t-SNE to train a model or interpreting its axes as meaningful units. The axes have no direct meaning.
