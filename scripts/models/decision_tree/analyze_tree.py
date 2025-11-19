import numpy as np

def analyze_tree_complexity(model, logger=None):
    """
    Analyze decision tree complexity metrics to detect overfitting.
    
    Parameters:
    -----------
    model : DecisionTreeClassifier or Pipeline
        Trained Decision Tree model or pipeline.
    logger : Logger or None
        Optional logger instance.
    
    Returns:
    --------
    complexity : dict
        Dictionary containing tree complexity metrics.
    """
    # Extract DT model if it's in a pipeline
    if hasattr(model, 'named_steps'):
        dt_model = model.named_steps['dt']
    else:
        dt_model = model
    
    tree = dt_model.tree_
    
    complexity = {
        'n_nodes': tree.node_count,
        'n_leaves': tree.n_leaves,
        'max_depth': tree.max_depth,
        'n_features_used': len(np.unique(tree.feature[tree.feature >= 0]))
    }
    
    if logger is not None:
        logger.info("=" * 50)
        logger.info("TREE COMPLEXITY ANALYSIS")
        logger.info("=" * 50)
        logger.info(f"Total nodes: {complexity['n_nodes']}")
        logger.info(f"Leaf nodes: {complexity['n_leaves']}")
        logger.info(f"Internal nodes: {complexity['n_nodes'] - complexity['n_leaves']}")
        logger.info(f"Max depth: {complexity['max_depth']}")
        logger.info(f"Features used: {complexity['n_features_used']}")
        
        # Warning for very complex trees
        if complexity['n_nodes'] > 1000:
            logger.warning("⚠️  Very large tree detected (>1000 nodes). "
                         "Consider pruning or using Random Forest.")
        if complexity['max_depth'] > 25:
            logger.warning("⚠️  Very deep tree detected (>25 levels). "
                         "May indicate overfitting.")
    
    return complexity