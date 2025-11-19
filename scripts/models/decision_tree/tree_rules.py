from sklearn.tree import export_text

def get_tree_rules(model, feature_names, max_depth=None):
    """
    Extract decision rules from the tree in text format.
    
    Parameters:
    -----------
    model : DecisionTreeClassifier or Pipeline
        Trained Decision Tree model or pipeline.
    feature_names : list
        List of feature names.
    max_depth : int or None
        Maximum depth to display. None for full tree.
    
    Returns:
    --------
    rules : str
        Text representation of decision rules.
    """
    # Extract DT model if it's in a pipeline
    if hasattr(model, 'named_steps'):
        dt_model = model.named_steps['dt']
    else:
        dt_model = model
    
    rules = export_text(dt_model, feature_names=feature_names, max_depth=max_depth)
    return rules