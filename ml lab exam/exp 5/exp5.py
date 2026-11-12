# Experiment 5: PCA + t-SNE + Random Forest (Education Data)
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

DATA = Path(__file__).resolve().parents[2] / "exp5" / "Bachelors or Higher Degree for USA States.csv"
df = pd.read_csv(DATA)
df.columns = df.columns.str.strip()          # remove leading spaces in column names

# ---------- 1. Build features + target ----------
features = df.select_dtypes("number")
target = "New Hampshire"
y = (features[target] >= features[target].median()).astype(int)
X = features.drop(columns=target)
print("SIZE:", X.shape)
print("Target counts:", y.value_counts().to_dict())

# ---------- 2. Split + scale ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------- 3. Baseline classifier ----------
rf = RandomForestClassifier(random_state=42).fit(X_train_s, y_train)
print("Baseline accuracy:", rf.score(X_test_s, y_test))

# ---------- 4. PCA vs t-SNE visualization ----------
X_pca = PCA(n_components=2).fit_transform(X_train_s)
X_tsne = TSNE(n_components=2, perplexity=5, random_state=42,
              max_iter=1000).fit_transform(X_train_s)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, data, title in [(axes[0], X_pca, "PCA"), (axes[1], X_tsne, "t-SNE")]:
    ax.scatter(data[:, 0], data[:, 1], c=y_train, cmap="coolwarm", edgecolor="k")
    ax.set_title(title)
plt.show()

# ---------- 5. PCA keeping 90% variance ----------
pca90 = PCA(n_components=0.90)
X_train_p = pca90.fit_transform(X_train_s)
X_test_p = pca90.transform(X_test_s)
rf_pca = RandomForestClassifier(random_state=42).fit(X_train_p, y_train)
print("PCA components:", pca90.n_components_)
print("PCA accuracy:", rf_pca.score(X_test_p, y_test))
