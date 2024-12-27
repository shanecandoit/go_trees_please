
# Trees please

It's hard to understand things sometimes.

Why trees?
Nueral networks work, but can't explain.
Trees explain but don't work.
But have we tried hard enough?

Assume infinite memory.
What would have this instance done? That instance?
We can know.
If we synthesize we can go forever.

Inspiration:

- [scikit-learn Decision Trees](https://scikit-learn.org/1.5/modules/tree.html)

## Decision Trees

Single Tree: A decision tree is a single model that makes decisions by splitting data based on feature values.
Interpretable: Decision trees are relatively easy to understand and visualize. You can follow the branches to see how a decision is made.
Prone to Overfitting: A single tree can be overly complex and sensitive to the training data, leading to poor performance on new data.

## Random Forests

Ensemble of Trees: A random forest is an ensemble method that combines multiple decision trees.
Reduced Overfitting: By averaging the predictions of many trees, random forests are less prone to overfitting and tend to be more accurate.
Less Interpretable: While individual trees are interpretable, the overall model is more complex and harder to understand.
Robustness: Random forests are often more robust to noise and outliers in the data.

### Deciding When to Branch or Start a New Tree

In decision tree algorithms, the decision of whether to start a new tree or add a branch to an existing one is typically determined by a combination of factors, including:

Tree Depth:

Maximum Depth: If a tree reaches a predefined maximum depth, it stops growing further. Any new data would then be used to train a new tree.
Minimum Node Size: If a node's sample size falls below a certain threshold, it's not split further, and a leaf node is created. This prevents overfitting.
Impurity Measure:

Algorithms like ID3, C4.5, and CART use impurity measures like information gain, gain ratio, or Gini impurity to determine the best split at each node.
If the best split doesn't significantly reduce impurity, the node may become a leaf node, or a new tree might be started depending on other factors.
Ensemble Methods:

In ensemble methods like Random Forest, each tree is trained on a random subset of the data. This introduces diversity and reduces overfitting.
New trees are continually added to the ensemble until a specific number of trees or a desired performance level is reached.
Specific Algorithms and Their Approaches:

ID3: Uses information gain to select the best attribute for splitting.
C4.5: Extends ID3 by using gain ratio to handle attributes with many values.
CART: Uses Gini impurity to measure the homogeneity of a node.
Random Forest: Builds multiple decision trees, each trained on a random subset of the data and features.

### Random Forest (RF) model would typically encapsulate the core functionalities and properties of the algorithm

Here's a breakdown of the potential methods and properties:

Properties:

n_estimators: The number of trees in the forest.
max_depth: The maximum depth of each tree.
min_samples_split: The minimum number of samples required to split an internal node.
min_samples_leaf: The minimum number of samples required to be at a leaf node.
max_features: The number of features to consider when looking for the best split.
criterion: The function to measure the quality of a split (e.g., 'gini', 'entropy').
random_state: Controls the randomness of the algorithm.

Methods:

fit(X, y): Trains the model on the training data X with target labels y.
predict(X): Predicts the class labels or regression values for the input data X.
predict_proba(X): Predicts class probabilities for each class in the case of classification problems.
score(X, y): Evaluates the model's performance on a given test set X with target labels y.
get_params(): Returns the parameters for the estimator.
set_params(): Sets the parameters of the estimator.
Additional Considerations:

Feature Importance: The RF class can provide feature importance scores, indicating which features are most influential in the model's predictions.
Hyperparameter Tuning: The class can incorporate methods for tuning hyperparameters like grid search or randomized search.
Visualization: The class can include methods for visualizing the decision trees within the forest.
Parallelism: The class can leverage parallel processing to speed up training and prediction.

## Notes on visualization

Install graphviz `scoop install main/graphviz`

This code will turn trees into dot file

```py
# grab the DOT code of the decision tree
from sklearn.tree import export_graphviz
export_graphviz(tree_clf, out_file="tree.dot", feature_names=iris.feature_names, class_names=iris.target_names, rounded=True, filled=True)
```

This code will turn that dot file into a pretty picture.

```bash
dot -Tpng tree.dot -o tree.png
```

### Gini

Gini Impurity is a measure of how pure a node is in a decision tree. In simpler terms, it tells us how likely it is that two randomly selected items from the same node will belong to different classes.

Key Points:

High Gini Impurity: Means the node contains a mix of different classes, making it impure.
Low Gini Impurity: Means the node primarily contains items from a single class, making it pure.
Gini Impurity = 0: Indicates perfect purity, meaning all items in the node belong to the same class.
Gini Impurity = 1: Indicates maximum impurity, meaning the node contains an equal proportion of all classes.

```py
import numpy as np
def gini(y):
  """Calculate the Gini impurity of a list of labels.

  Args:
    y: A list of labels.

  Returns:
    The Gini impurity of the list.
  """

  if len(y) == 0:
    return 0

  unique_labels, counts = np.unique(y, return_counts=True)
  probabilities = counts / len(y)
  return 1 - np.sum(probabilities ** 2)
```

---

What do we want to do?

iris data set looks like

```csv
sepal length (cm),sepal width (cm),petal length (cm),petal width (cm),target
5.0,3.3,1.4,0.2,0
7.0,3.2,4.7,1.4,1
5.7,2.8,4.1,1.3,1
6.3,3.3,6.0,2.5,2
```
