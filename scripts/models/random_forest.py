"""
Random Forest Classifier Module

Provides functions specific to Random Forest training and configuration.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif


def create_rf_pipeline(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=0,
    use_smote=True,
    use_scaler=True,
    use_feature_selection=False,
    n_features=30,
    class_weight='balanced'
):
    """
    Create a Random Forest pipeline with preprocessing steps.
    
    Parameters:
    -----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int, default=10
        Maximum depth of each tree.
    min_samples_split : int, default=5
        Minimum samples required to split an internal node.
    min_samples_leaf : int, default=2
        Minimum samples required to be at a leaf node.
    max_features : str, int, float or None, default='sqrt'
        Number of features to consider when looking for the best split.
    random_state : int, default=0
        Random state for reproducibility.
    use_smote : bool, default=True
        Whether to apply SMOTE oversampling.
    use_scaler : bool, default=True
        Whether to apply StandardScaler.
    use_feature_selection : bool, default=False
        Whether to apply feature selection.
    n_features : int, default=30
        Number of features to select (if use_feature_selection=True).
    class_weight : str or dict, default='balanced'
        Class weights for the Random Forest.
    
    Returns:
    --------
    pipeline : ImbPipeline
        Complete preprocessing and training pipeline.
    """
    steps = []
    
    # Add StandardScaler
    if use_scaler:
        steps.append(('scaler', StandardScaler()))
    
    # Add feature selection
    if use_feature_selection:
        steps.append(('feature_selection', SelectKBest(f_classif, k=n_features)))
    
    # Add SMOTE
    if use_smote:
        steps.append(('smote', SMOTE(sampling_strategy='auto', random_state=random_state)))
    
    # Add Random Forest
    steps.append(('rf', RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1
    )))
    
    return ImbPipeline(steps)


def train_random_forest(
    X_train,
    y_train,
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=0,
    cv=5,
    class_weight='balanced',
    logger=None
):
    """
    Train and evaluate a Random Forest classifier with cross-validation.

    Parameters:
    -----------
    X_train : array-like
        Feature matrix for training.
    y_train : array-like
        Target vector for training.
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int, default=10
        Maximum depth of each tree.
    min_samples_split : int, default=5
        Minimum samples required to split an internal node.
    min_samples_leaf : int, default=2
        Minimum samples required to be at a leaf node.
    max_features : str, int, float or None, default='sqrt'
        Number of features to consider when looking for the best split.
    random_state : int, default=0
        Controls the randomness for reproducibility.
    cv : int, default=5
        Number of folds for cross-validation.
    class_weight : str or dict, default='balanced'
        Weights associated with classes.
    logger : Logger or None, default=None
        Optional logger instance for logging messages.

    Returns:
    --------
    rf : RandomForestClassifier
        The trained RandomForestClassifier instance.
    cv_scores : ndarray
        Array of cross-validation scores.
    """
    # Initialize Random Forest
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1
    )
    
    # Perform cross-validation
    if logger is not None:
        logger.info("Performing Random Forest cross-validation...")
    
    cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, n_jobs=-1)
    
    # Train the model on full training set
    if logger is not None:
        logger.info("Training Random Forest on full training set...")
    
    rf.fit(X_train, y_train)
    
    # Log results
    if logger is not None:
        logger.info('=' * 50)
        logger.info('RANDOM FOREST MODEL')
        logger.info('=' * 50)
        logger.info(f'Parameters: n_estimators={n_estimators}, max_depth={max_depth}, '
                   f'max_features={max_features}')
        logger.info(f'Cross-validation scores: {cv_scores}')
        logger.info(f'Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')
    
    return rf, cv_scores


def tune_rf_hyperparameters(X_train, y_train, cv=5, n_jobs=-1, logger=None):
    """
    Perform grid search to find optimal Random Forest hyperparameters.
    
    Parameters:
    -----------
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.
    cv : int, default=5
        Number of cross-validation folds.
    n_jobs : int, default=-1
        Number of parallel jobs.
    logger : Logger or None
        Optional logger instance.
    
    Returns:
    --------
    best_params : dict
        Best hyperparameters found.
    best_score : float
        Best CV score achieved.
    grid_search : GridSearchCV
        Fitted GridSearchCV object.
    """
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    if logger:
        logger.info("Starting Random Forest grid search...")
        logger.info(f"Parameter grid: {param_grid}")
    
    # Create Random Forest classifier
    rf = RandomForestClassifier(random_state=0, class_weight='balanced', n_jobs=-1)
    
    # Perform grid search
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=cv,
        n_jobs=n_jobs,
        scoring='accuracy',
        verbose=1 if logger else 0
    )
    
    grid_search.fit(X_train, y_train)
    
    if logger:
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_, grid_search.best_score_, grid_search


def get_rf_oob_score(X_train, y_train, n_estimators=100, max_depth=10, 
                     random_state=0, logger=None):
    """
    Train Random Forest with out-of-bag scoring.
    
    OOB score is a built-in cross-validation for Random Forest.
    
    Parameters:
    -----------
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.
    n_estimators : int, default=100
        Number of trees.
    max_depth : int, default=10
        Maximum depth.
    random_state : int, default=0
        Random state.
    logger : Logger or None
        Optional logger.
    
    Returns:
    --------
    rf : RandomForestClassifier
        Trained model with OOB score.
    oob_score : float
        Out-of-bag score.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        oob_score=True,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    
    if logger:
        logger.info(f"Random Forest OOB Score: {rf.oob_score_:.4f}")
    
    return rf, rf.oob_score_