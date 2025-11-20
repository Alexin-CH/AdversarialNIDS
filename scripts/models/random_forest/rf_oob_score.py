from sklearn.ensemble import RandomForestClassifier

def get_rf_oob_score(X_train, y_train, n_estimators=100, max_depth=10, 
                     random_state=0, logger=None):
    """
    Train Random Forest with out-of-bag scoring.
    OOB score is a built-in cross-validation for Random Forest.
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