# Experiment 3 — Decision Tree Classification + Pruning (Calorie Efficiency)

## Aim
Classify `calorie_efficiency` (Low / Moderate / High) using a decision tree, then compare an unpruned tree with a depth-limited tree and a pruned tree to understand overfitting.

## Dataset
`exp 3/calorie_efficiency_dataset.csv` — full file has **1,000,000 rows** × 15 columns. The script reads the first **30,000 rows** for speed. No missing values.

## Libraries
`pandas`, `matplotlib`, `scikit-learn` (`train_test_split`, `LabelEncoder`, `DecisionTreeClassifier`, `plot_tree`, `ConfusionMatrixDisplay`).

## Steps
1. Load a sample and inspect `shape`, `info()`, `describe()`, missing values, and class counts.
2. Encode the target labels with `LabelEncoder`.
3. Split 70/30 with `stratify=y`.
4. Train three trees:
   - **Unpruned:** `DecisionTreeClassifier(random_state=42)`
   - **Depth-limited:** `max_depth=5`
   - **Pruned:** `ccp_alpha=0.01`
5. Compare **train accuracy vs test accuracy** — a large gap means overfitting.
6. Evaluate the depth-limited tree with a classification report and confusion matrix.
7. Plot the top 3 levels of the tree with `plot_tree`.

## Key concepts
- **Gini impurity** measures how mixed a node is; the tree splits to reduce it.
- **Unpruned tree** keeps splitting until leaves are pure → training accuracy 1.0, but it memorizes noise.
- **`max_depth`** limits tree size (pre-pruning).
- **`ccp_alpha`** is cost-complexity pruning (post-pruning): larger values prune more branches.

## Expected output
```
Unpruned: train=1.0000, test=0.8827
Depth=5 : train=0.9497, test=0.9486
Pruned  : train=0.9492, test=0.9491
```
Overall accuracy ≈ **0.95**, but `High Efficiency` recall is **0.00** because that class is only ~2.5% of the data. This is the **accuracy paradox** — always check per-class precision/recall.

## Viva questions
1. **Why prune a decision tree?** To reduce overfitting and improve generalization.
2. **What is `ccp_alpha`?** Cost-complexity pruning parameter; higher values produce a smaller tree.
3. **Why is unpruned training accuracy 1.0?** The tree can keep splitting until every training point is isolated.
4. **Why is accuracy misleading here?** The classes are imbalanced; a model can predict “Low” almost always and still score ~94%.
5. **How can you handle imbalance?** `class_weight="balanced"`, resampling, or evaluate with macro-F1/recall.

## Common mistake
Reporting only accuracy on an imbalanced dataset. Also, loading the full 1M rows in an exam can be slow — use `nrows=30000` for a quick demo.
