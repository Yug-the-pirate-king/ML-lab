# Experiment 6 — Recommender System (State Similarity)

## Aim
Build a simple **state-to-state recommender** using cosine and Pearson similarity, recommend states similar to California, and predict California's latest education value.

## Dataset
`exp6/Bachelors or Higher Degree for USA States.csv` — 16 years × 52 columns (date + 50 states + District of Columbia), no missing values. After setting the date as the index, the matrix is 16 × 51 (years × jurisdictions).

## Libraries
`pandas`, `numpy`, `matplotlib`, `scikit-learn` (`cosine_similarity`).

## Steps
1. Load the CSV, strip spaces from column names, set the date column as the index.
2. Fill any missing values with the column mean.
3. Transpose the matrix (`df.T`) so each row is a **state** and each column is a year.
4. Compute **cosine similarity** between states, set the diagonal to 0, and list the top 5 states similar to California.
5. Predict California's latest value as a similarity-weighted average of the other states.
6. Compute **Pearson correlation** between states and compare the top matches.
7. Plot a heatmap of the top similar states.

## Key code
```python
sim = cosine_similarity(df.T)          # states × states
np.fill_diagonal(sim, 0)               # ignore self-similarity
sim_df = pd.DataFrame(sim, index=df.columns, columns=df.columns)
print(sim_df["California"].sort_values(ascending=False).head())
pred = (df.loc[year] * sim_df["California"]).sum() / sim_df["California"].sum()
```

## Expected output
Top cosine matches for California:
```
Massachusetts, New York, Virginia, Illinois, Colorado
```
- Actual California 2021 value: **36.20**
- Predicted value: **34.63**

Top Pearson matches:
```
Texas, Massachusetts, New Jersey, Georgia, North Carolina
```

## Viva questions
1. **User-based vs item-based collaborative filtering?** User-based finds similar users and recommends their items; item-based finds similar items and recommends them.
2. **Cosine vs Pearson?** Cosine measures the angle between vectors (magnitude-insensitive); Pearson measures linear correlation after centering.
3. **Why set the diagonal to 0?** A state is perfectly similar to itself; keeping it would dominate the recommendation.
4. **What is the cold-start problem?** A new user/item has no history, so similarity cannot be computed.
5. **How do you evaluate a recommender?** MAE/RMSE on held-out ratings, precision@k, recall@k, or ranking metrics.

## Common mistake
Computing similarity between **years** instead of **states**. Transpose the matrix so each row is a state before comparing.
