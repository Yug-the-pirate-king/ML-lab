# Experiment 6: Recommender System (Cosine / Pearson Similarity)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path(__file__).resolve().parents[2] / "exp6" / "Bachelors or Higher Degree for USA States.csv"
df = pd.read_csv(DATA)
df.columns = df.columns.str.strip()
df = df.set_index("Unnamed: 0").fillna(df.mean(numeric_only=True))

# ---------- 1. Data ----------
print("SIZE:", df.shape)                     # years x states
print("MISSING:", df.isnull().sum().sum())

# ---------- 2. State-to-state cosine similarity ----------
sim = cosine_similarity(df.T)                # compare states, not years
np.fill_diagonal(sim, 0)                     # ignore self-similarity
sim_df = pd.DataFrame(sim, index=df.columns, columns=df.columns)

target = "California"
print("Top 5 states similar to California (cosine):")
print(sim_df[target].sort_values(ascending=False).head())

# ---------- 3. Predict California's latest value ----------
year = df.index[-1]
weights = sim_df[target]
pred = (df.loc[year] * weights).sum() / weights.sum()
print(f"Actual {year} California: {df.loc[year, target]:.2f}")
print(f"Predicted: {pred:.2f}")

# ---------- 4. Pearson similarity ----------
pearson = df.corr()
print("Top 5 states similar to California (Pearson):")
print(pearson[target].drop(target).sort_values(ascending=False).head())

# ---------- 5. Heatmap of the top states ----------
top = sim_df[target].sort_values(ascending=False).head(6).index
plt.imshow(df[top].T.corr(), cmap="coolwarm")
plt.xticks(range(len(top)), top, rotation=90)
plt.yticks(range(len(top)), top)
plt.colorbar(label="Pearson correlation")
plt.title("State similarity")
plt.tight_layout()
plt.show()
