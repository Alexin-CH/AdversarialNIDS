def estimate_knn_memory(n_samples, n_features, logger=None):
    """
    Estimate memory usage for KNN model.
    
    KNN stores all training data, so memory can be significant.
    
    Parameters:
    -----------
    n_samples : int
        Number of training samples.
    n_features : int
        Number of features.
    logger : Logger or None
        Optional logger.
    
    Returns:
    --------
    memory_mb : float
        Estimated memory in megabytes.
    """
    # Assuming float64 (8 bytes per value)
    bytes_per_value = 8
    total_bytes = n_samples * n_features * bytes_per_value
    memory_mb = total_bytes / (1024 ** 2)
    
    if logger:
        logger.info(f"Estimated KNN memory usage: {memory_mb:.2f} MB")
        logger.info(f"  ({n_samples} samples × {n_features} features × {bytes_per_value} bytes)")
    
    return memory_mb