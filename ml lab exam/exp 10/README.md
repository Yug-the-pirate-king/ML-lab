# Experiment 10 — Perceptron from Scratch + MLP (Neural Networks)

## Aim
Implement a **single-layer perceptron** from scratch with NumPy, then train a **Multilayer Perceptron (MLP)** on non-linear data and compare activation functions and hidden-layer sizes.

## Dataset
- **Perceptron:** `make_blobs` — 100 points, 2 linearly separable classes.
- **MLP:** `make_moons` — 300 points, 2 non-linear classes (crescent shapes).
No external CSV needed.

## Libraries
`numpy`, `matplotlib`, `scikit-learn` (`make_blobs`, `make_moons`, `train_test_split`, `MLPClassifier`, `accuracy_score`).

## Steps
1. Generate linearly separable blobs and convert labels to 0/1.
2. Implement the perceptron:
   - initialize weights and bias to 0
   - compute `z = w·x + b`
   - predict 1 if `z ≥ 0`, else 0
   - update: `w += lr × (y − ŷ) × x`, `b += lr × (y − ŷ)`
3. Train it and plot the decision boundary.
4. Generate the non-linear moons dataset, split 70/30, and train `MLPClassifier` with `logistic`, `tanh`, and `relu` activations.
5. Compare hidden-layer sizes `(8, 4)` and `(64, 32)`.

## Key code
```python
y_hat = 1 if xi @ self.w + self.b >= 0 else 0
update = self.lr * (yi - y_hat)
self.w += update * xi
self.b += update

mlp = MLPClassifier(hidden_layer_sizes=(50,), activation="relu", max_iter=1000)
mlp.fit(X_train, y_train)
print(mlp.score(X_test, y_test))
```

## Expected output
```
Perceptron accuracy: 1.0
logistic: 0.8222
tanh: 0.8556
relu: 0.8889
layers=(8, 4): 0.8556
layers=(64, 32): 0.9222
```

## Viva questions
1. **Why can a single perceptron not solve XOR/moons?** It can only draw a straight line (linear boundary); moons need a non-linear boundary.
2. **What does an activation function do?** Adds non-linearity so stacked layers can learn complex boundaries.
3. **Why is ReLU popular?** It is cheap to compute and reduces the vanishing-gradient problem.
4. **What is backpropagation?** The algorithm that computes gradients of the loss with respect to weights using the chain rule.
5. **What happens if you add too many layers/neurons?** The model can overfit; use validation accuracy and regularization.
6. **Why scale inputs for an MLP?** Gradient descent converges faster when features have similar ranges.

## Common mistake
Forgetting `return self` in a custom `fit` method, so the trained model object is `None`. Also, using a linear activation in every layer turns the network back into a linear model.
