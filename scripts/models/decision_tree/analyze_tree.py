import numpy as np

def analyze_tree_complexity(dt_model, logger=None):
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
    return complexity