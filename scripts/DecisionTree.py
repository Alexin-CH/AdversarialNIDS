# Decision Tree Classifier

"""
A decision tree algorithm is a supervised machine learning method that models decisions and their possible consequences as a tree-like structure, where each internal node represents a test on a feature, each branch represents an outcome, and each leaf node represents a class label or regression value.

Parameters:
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from logger import LoggerManager

def train_decision_tree(X_train, y_train, max_depth=6, logger=None):
    decisionTree = DecisionTreeClassifier(max_depth=max_depth)
    decisionTree.fit(X_train, y_train)
    cv_decisionTree = cross_val_score(decisionTree, X_train, y_train, cv=5)
    if logger is not None:
        logger.info('Decision Tree Classifier trained successfully.')
        logger.info('\nCross-validation scores: %s', ', '.join(map(str, cv_decisionTree)))
        logger.info('\nMean cross-validation score: %.2f', cv_decisionTree.mean())
    return
