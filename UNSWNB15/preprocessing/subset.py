import numpy as np

def subset_indices(data, size=None, logger=None):
    """ Undersamples the dataset to the specified size. """
    unique, counts = np.unique(data, return_counts=True)

    if size is None:
        size = np.max(counts) * len(unique)
        if logger:
            logger.info(f"No size specified. Using maximum class size: {size}")

    max_size = size // len(unique)

    if logger:
        # Display num of elements per class
        logger.info("Class distribution before subsetting:")
        for i in range(len(unique)):
            logger.info(f"  Class {unique[i]}: {counts[i]} samples")

    # Undersample each class to the max_size
    subset_indices = []
    for cls in unique:
        cls_indices = np.where(data == cls)[0]
        if len(cls_indices) > max_size:
            cls_indices = np.random.choice(cls_indices, max_size, replace=False)
        subset_indices.extend(cls_indices)

    if logger:
        logger.info(f"Subsetted dataset to size: {len(subset_indices)}")
    return subset_indices
