# Experiment 10: Perceptron from Scratch + MLP (scikit-learn)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ---------- 1. Single-layer perceptron from scratch ----------
class Perceptron:
    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                y_hat = 1 if xi @ self.w + self.b >= 0 else 0
                update = self.lr * (yi - y_hat)
                self.w += update * xi
                self.b += update
        return self

    def predict(self, X):
        return (X @ self.w + self.b >= 0).astype(int)

X, y = make_blobs(n_samples=100, centers=2, n_features=2,
                  random_state=42, cluster_std=1.2)
y = (y > 0).astype(int)
p = Perceptron().fit(X, y)
print("Perceptron accuracy:", accuracy_score(y, p.predict(X)))

# decision boundary
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z = p.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.3, cmap="bwr")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolor="k")
plt.title("Perceptron decision boundary")
plt.show()

# ---------- 2. MLP on non-linear moons data ----------
X, y = make_moons(n_samples=300, noise=.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

for act in ["logistic", "tanh", "relu"]:
    mlp = MLPClassifier(hidden_layer_sizes=(50,), activation=act,
                        max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    print(f"{act}: {mlp.score(X_test, y_test):.4f}")

# ---------- 3. Effect of hidden-layer size ----------
for size in [(8, 4), (64, 32)]:
    mlp = MLPClassifier(hidden_layer_sizes=size, max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    print(f"layers={size}: {mlp.score(X_test, y_test):.4f}")
