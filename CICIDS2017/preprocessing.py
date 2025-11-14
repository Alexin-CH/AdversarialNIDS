import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

import kagglehub


def preprocess_cicids2017(logger=None):
    """
    Preprocess the CICIDS2017 dataset from Kaggle.
    
    This function downloads the CICIDS2017 dataset, performs data cleaning including:
    - Removing duplicate rows
    - Handling missing values
    - Replacing infinite values
    - Filling specific columns with median values
    
    Args:
        logger: Optional logger instance for tracking preprocessing steps
    
    Returns:
        pd.DataFrame: Preprocessed dataset
    
    Raises:
        FileNotFoundError: If the dataset file cannot be found
        ValueError: If the dataset is empty or invalid
    """
    
    try:
        # Download dataset
        if logger:
            logger.info("Downloading dataset: sweety18/cicids2017-full-dataset")
        
        path_cicids2017 = kagglehub.dataset_download("sweety18/cicids2017-full-dataset")
        file_path = os.path.join(path_cicids2017, "combine.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found at: {file_path}")
        
        if logger:
            logger.info(f"Loading data from: {file_path}")
        
        data = pd.read_csv(file_path, low_memory=False)
        
        if data.empty:
            raise ValueError("Dataset is empty")
        
        # Initial dimensions
        rows, cols = data.shape
        
        if logger:
            logger.info(f"Initial dimensions: {rows:,} rows x {cols} columns = {rows * cols:,} cells")
        
        # Strip whitespace from column names
        if logger:
            logger.info("Cleaning column names")
        
        col_names = {col: col.strip() for col in data.columns}
        data.rename(columns=col_names, inplace=True)
        
        # Remove duplicates
        if logger:
            logger.info("Removing duplicate rows")
        
        initial_rows = len(data)
        data.drop_duplicates(inplace=True)
        duplicates_removed = initial_rows - len(data)
        
        if logger:
            logger.info(f"Removed {duplicates_removed:,} duplicate rows. Remaining: {len(data):,}")
        
        # Remove rows with any missing values (initial pass)
        if logger:
            logger.info("Removing rows with missing values (initial pass)")
        
        initial_rows = len(data)
        data.dropna(inplace=True)
        na_removed = initial_rows - len(data)
        
        if logger:
            logger.info(f"Removed {na_removed:,} rows with missing values. Remaining: {len(data):,}")
        
        # Handle infinity values
        if logger:
            logger.info("Checking for infinite values in numeric columns")
        
        numeric_cols = data.select_dtypes(include=np.number).columns
        
        # Check for infinity values before replacement
        inf_mask = np.isinf(data[numeric_cols])
        inf_count = inf_mask.sum()
        
        if inf_count.sum() > 0 and logger:
            logger.info("Columns with infinite values:")
            for col, count in inf_count[inf_count > 0].items():
                logger.info(f"  {col}: {count:,} infinite values")
        
        # Replace infinite values with NaN
        initial_missing = data.isna().sum().sum()
        
        if logger:
            logger.info(f"Missing values before processing infinite values: {initial_missing:,}")
        
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        final_missing = data.isna().sum().sum()
        inf_converted = final_missing - initial_missing
        
        if logger:
            logger.info(f"Missing values after processing infinite values: {final_missing:,}")
            logger.info(f"Infinite values converted to NaN: {inf_converted:,}")
        
        # Analyze missing values
        missing = data.isna().sum()
        missing_cols = missing[missing > 0]
        
        if len(missing_cols) > 0 and logger:
            mis_per = (missing / len(data)) * 100
            logger.info("Columns with missing values:")
            for col in missing_cols.index:
                count = missing[col]
                percentage = mis_per[col]
                logger.info(f"  {col}: {count:,} ({percentage:.2f}%)")
        
        # Handle specific columns with median imputation
        flow_cols = ['Flow Bytes/s', 'Flow Packets/s']
        
        for col in flow_cols:
            if col in data.columns:
                median_value = data[col].median()
                missing_count = data[col].isna().sum()
                
                if missing_count > 0:
                    if logger:
                        logger.info(f"Filling {missing_count:,} missing values in '{col}' with median: {median_value:.2f}")
                    
                    data.fillna({col: median_value}, inplace=True)
        
        # Verify no missing values remain in flow columns
        if logger:
            for col in flow_cols:
                if col in data.columns:
                    remaining_missing = data[col].isna().sum()
                    logger.info(f"Remaining missing values in '{col}': {remaining_missing}")
        
        # Final dimensions
        final_rows, final_cols = data.shape
        total_removed = rows - final_rows
        retention_rate = (final_rows / rows * 100)
        
        if logger:
            logger.info("=" * 60)
            logger.info("Preprocessing completed successfully")
            logger.info(f"Final dimensions: {final_rows:,} rows × {final_cols} columns")
            logger.info(f"Total rows removed: {total_removed:,} ({(total_removed / rows * 100):.2f}%)")
            logger.info(f"Data retention rate: {retention_rate:.2f}%")
            logger.info("=" * 60)
        
        return data
    
    except FileNotFoundError as e:
        if logger:
            logger.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError:
        error_msg = "The CSV file is empty"
        if logger:
            logger.error(error_msg)
        raise ValueError(error_msg)
    except Exception as e:
        if logger:
            logger.error(f"Error during preprocessing: {e}")
        raise

def data_encoding(data, logger=None):
    """
    Encode the 'Label' column of the dataset into attack types and numerical labels.
    Args:
        data (pd.DataFrame): The DataFrame containing the 'Label' column.
        logger: Optional logger instance for tracking encoding steps.
    Returns:
        pd.DataFrame: The DataFrame with encoded attack types and numerical labels.
    """
    if logger:
        logger.info("Starting data encoding process")
        logger.info(f"Data Labels before encoding: {data['Label'].value_counts()}")

    try:
        # Creating a dictionary that maps each label to its attack type
        attack_map = {
            'BENIGN': 'BENIGN',
            'DDoS': 'DDoS',
            'DoS Hulk': 'DoS',
            'DoS GoldenEye': 'DoS',
            'DoS slowloris': 'DoS',
            'DoS Slowhttptest': 'DoS',
            'PortScan': 'Port Scan',
            'FTP-Patator': 'Brute Force',
            'SSH-Patator': 'Brute Force',
            'Bot': 'Bot',
            'Web Attack Brute Force': 'Web Attack',
            'Web Attack XSS': 'Web Attack',
            'Web Attack Sql Injection': 'Web Attack',
            'Infiltration': 'Infiltration',
            'Heartbleed': 'Heartbleed'
        }

        # Creating a new column 'Attack Type' in the DataFrame based on the attack_map dictionary
        data['Attack Type'] = data['Label'].map(attack_map)

        data.drop('Label', axis = 1, inplace = True)

        le = LabelEncoder()
        data['Attack Number'] = le.fit_transform(data['Attack Type'])

        if logger:
            logger.info(f"Data Labels after encoding: {data['Attack Type'].value_counts()}")

    except KeyError as e:
        if logger:
            logger.error(f"KeyError during data encoding: {e}")
        raise

    return data

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