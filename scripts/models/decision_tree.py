from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

def train_decision_tree(X_train, y_train, max_depth=6, logger=None):
    decisionTree = DecisionTreeClassifier(max_depth=max_depth)
    decisionTree.fit(X_train, y_train)
    cv_decisionTree = cross_val_score(decisionTree, X_train, y_train, cv=5)
    if logger is not None:
        logger.info('Decision Tree Classifier trained successfully.')
        logger.info('\nCross-validation scores: %s', ', '.join(map(str, cv_decisionTree)))
        logger.info('\nMean cross-validation score: %.2f', cv_decisionTree.mean())
    return decisionTree, cv_decisionTree
