# Experiment 2: Simple & Multiple Linear Regression + Residuals
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

DATA = Path(__file__).resolve().parents[2] / "exp -2" / "employee_salary_regression.csv"
df = pd.read_csv(DATA)

# ---------- 1. Data analysis ----------
print("SIZE:", df.shape)
print(df.info())
print(df.describe())
print("MISSING:", df.isnull().sum().sum())

# ---------- 2. Simple linear regression: y = mx + b ----------
x = df["years_experience"]
y = df["annual_salary_usd"]
m, b = np.polyfit(x, y, 1)
print(f"Salary = {m:.2f} * experience + {b:.2f}")

plt.scatter(x, y, alpha=.5)
plt.plot(x, m * x + b, color="red")
plt.xlabel("Years of experience")
plt.ylabel("Annual salary (USD)")
plt.title("Simple Linear Regression")
plt.show()

# ---------- 3. Multiple linear regression ----------
features = ["years_experience", "age", "performance_score", "num_skills"]
X = df[features]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
residuals = y_test - pred

print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("R2  :", r2_score(y_test, pred))
print("Coefficients:", dict(zip(features, model.coef_.round(2))))

# ---------- 4. Residual plot ----------
plt.scatter(pred, residuals, alpha=.7)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted salary")
plt.ylabel("Residual (actual - predicted)")
plt.title("Residual Plot")
plt.show()
