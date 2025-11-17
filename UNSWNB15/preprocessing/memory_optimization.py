import numpy as np
import pandas as pd

def optimize_memory_usage(data, logger=None):
    """
    Optimize memory usage of a pandas DataFrame by downcasting numeric columns.
    Args:
        data (pd.DataFrame): The DataFrame to optimize.
    Returns:
        pd.DataFrame: The optimized DataFrame.
    """
    # For improving performance and reduce memory-related errors
    old_memory_usage = data.memory_usage().sum() / 1024 ** 2
    if logger:
        logger.info(f'Initial memory usage: {old_memory_usage:.2f} MB')
    for col in data.columns:
        col_type = data[col].dtype
        if col_type != object:
            c_min = data[col].min()
            c_max = data[col].max()
            # Downcasting float64 to float32
            if str(col_type).find('float') >= 0 and c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                data[col] = data[col].astype(np.float32)

            # Downcasting int64 to int32
            elif str(col_type).find('int') >= 0 and c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                data[col] = data[col].astype(np.int32)

    new_memory_usage = data.memory_usage().sum() / 1024 ** 2
    if logger:
        logger.info(f'Optimized memory usage: {new_memory_usage:.2f} MB')
        logger.info(f'Memory reduction: {old_memory_usage - new_memory_usage:.2f} MB ({((old_memory_usage - new_memory_usage) / old_memory_usage) * 100:.2f}%)')
    return data