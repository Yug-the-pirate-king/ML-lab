# Experiment 2 — Linear Regression + Residual Analysis (Salary)

## Aim
Predict `annual_salary_usd` from employee features using simple and multiple linear regression, then check the model with a residual plot.

## Dataset
`exp -2/employee_salary_regression.csv` — 1000 rows × 10 columns, no missing values. Target: `annual_salary_usd`.

## Libraries
`pandas`, `numpy`, `matplotlib`, `scikit-learn` (`train_test_split`, `LinearRegression`, `mean_squared_error`, `r2_score`).

## Steps
1. Load and inspect `shape`, `info()`, `describe()`, and missing values.
2. **Simple regression:** use `years_experience` only, fit a line with `np.polyfit(x, y, 1)`.
3. **Multiple regression:** use `years_experience`, `age`, `performance_score`, `num_skills`.
4. Split 80/20, fit `LinearRegression`, predict on the test set.
5. Compute **RMSE** and **R²**.
6. Plot **residual = actual − predicted** against predicted salary. The cloud should be random around 0.

## Key formulas
- Line: `y = mx + b`
- Residual: `e = y_actual − y_predicted`
- RMSE: `sqrt(mean(e²))` — average error in the same unit as salary.
- R²: fraction of the target's variance explained by the model (1 = perfect, 0 = no better than the mean).

## Key code
```python
m, b = np.polyfit(df["years_experience"], df["annual_salary_usd"], 1)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
residuals = y_test - pred
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("R2:", r2_score(y_test, pred))
```

## Expected output
- Simple line: `Salary = 2826.12 × experience + 84681.16`
- RMSE ≈ **8927**
- R² ≈ **0.939** (about 94% of salary variance explained)
- Residuals should be a random cloud around the red zero line.

## Viva questions
1. **What is a residual?** The difference between the actual and predicted value.
2. **What does a good residual plot look like?** Random scatter around 0 with equal spread — no curve or funnel.
3. **What does R² = 0.94 mean?** The model explains 94% of the variation in salary.
4. **Why split the data?** To test the model on unseen data and detect overfitting.
5. **Why not use `employee_id` as a feature?** It is an arbitrary identifier with no predictive meaning.

## Common mistake
Plotting `predicted − actual` instead of `actual − predicted`. The standard residual is `actual − predicted`.
