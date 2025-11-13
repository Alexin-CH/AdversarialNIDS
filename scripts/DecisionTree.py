# Decision Tree Classifier
"""
A decision tree algorithm is a supervised machine learning method that models decisions and their possible consequences as a tree-like structure, where each internal node represents a test on a feature, each branch represents an outcome, and each leaf node represents a class label or regression value.

Parameters:
- X_train: The feature matrix (input variables) used to train the decision tree.
- y_train: The target vector (labels) corresponding to X_train.
- max_depth=6: Sets the maximum depth of the decision tree to 6, controlling model complexity and helping prevent overfitting.
- cv=5: Specifies 5-fold cross-validation, meaning the data is split into 5 parts to evaluate the model’s performance more reliably.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

def train_decision_tree(X_train, y_train, max_depth=6):
    decisionTree = DecisionTreeClassifier(max_depth=max_depth)
    decisionTree.fit(X_train, y_train)
    cv_decisionTree = cross_val_score(decisionTree, X_train, y_train, cv=5)
    print('Decision Tree Classifier trained successfully.')
    print('\nCross-validation scores:', ', '.join(map(str, cv_decisionTree)))
    print(f'\nMean cross-validation score: {cv_decisionTree.mean():.2f}')
    return
