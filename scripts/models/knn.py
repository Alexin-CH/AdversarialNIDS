from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from scripts.logger import LoggerManager


def train_knn(
	X_train,
	y_train,
	n_neighbors=5,
	cv=5,
	logger=None
):
	"""
	Train and evaluate a K-Nearest Neighbors classifier with cross-validation.

	Parameters:
	- X_train: Feature matrix for training.
	- y_train: Target vector for training.
	- n_neighbors: Number of neighbors to use (default: 5).
	- cv: Number of folds for cross-validation (default: 5).
	- logger: Optional logger instance for logging messages. If None, uses print.

	Returns:
	- The trained KNeighborsClassifier instance.
	- The array of cross-validation scores.
	"""
	knn = KNeighborsClassifier(n_neighbors=n_neighbors)
	knn.fit(X_train, y_train)
	cv_scores = cross_val_score(knn, X_train, y_train, cv=cv)
	if logger is not None:
		logger.info('K-Nearest Neighbors Model')
		logger.info('\nCross-validation scores: %s', ', '.join(map(str, cv_scores)))
		logger.info('\nMean cross-validation score: %.2f', cv_scores.mean())
	return knn, cv_scores
