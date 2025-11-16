import os
import sys
import pandas as pd
import numpy as np  
import kagglehub

current_dir = os.getcwd()
sys.path.append(current_dir)

from scripts.logger import SimpleLogger

def download_prepare(logger=SimpleLogger(),flag=0):
    """ Download and prepare the corresponding dataset.
        Flag = 0 means cicids2017 and Flag = 1 means unsw-nb15 """
    try:
        # Download dataset
        dataset = ""
        name_csv = ""
        if flag ==0:
            dataset = "sweety18/cicids2017-full-dataset"
            name_csv ="combine.csv"
        elif flag==1 :
            dataset ="mrwellsdavid/unsw-nb15"
            name_csv = "UNSW-NB15_1.csv"
        else:
            logger.info("No corresponding Dataset") 
        logger.info("Downloading dataset: %s",dataset)
        path = kagglehub.dataset_download(dataset)
        file_path = os.path.join(path, name_csv)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found at: {file_path}")
        
        logger.info("Loading data")
        logger.debug(file_path)
        data = pd.read_csv(file_path, low_memory=False)
        
        if data.empty:
            raise ValueError("Dataset is empty")
        
        # Initial dimensions
        rows, cols = data.shape
        logger.info(f"Initial dimensions: {rows:,} rows x {cols} columns = {rows * cols:,} cells")
        
        # Strip whitespace from column names
        logger.debug("Cleaning column names")
        col_names = {col: col.strip() for col in data.columns}
        data.rename(columns=col_names, inplace=True)
        
        # Remove duplicates
        logger.debug("Removing duplicate rows")
        initial_rows = len(data)
        data.drop_duplicates(inplace=True)
        duplicates_removed = initial_rows - len(data)
        
        logger.debug(f"Removed {duplicates_removed:,} duplicate rows. Remaining: {len(data):,}")
        
        # Remove rows with any missing values (initial pass)
        logger.debug("Removing rows with missing values (initial pass)")
        initial_rows = len(data)
        data.dropna(inplace=True)
        na_removed = initial_rows - len(data)
        
        logger.debug(f"Removed {na_removed:,} rows with missing values. Remaining: {len(data):,}")
        
        # Handle infinity values
        logger.debug("Checking for infinite values in numeric columns")
        numeric_cols = data.select_dtypes(include=np.number).columns
        
        # Check for infinity values before replacement
        inf_mask = np.isinf(data[numeric_cols])
        inf_count = inf_mask.sum()
        
        if inf_count.sum() > 0 and logger:
            logger.debug("Columns with infinite values:")
            for col, count in inf_count[inf_count > 0].items():
                logger.debug(f"  {col}: {count:,} infinite values")
        
        # Replace infinite values with NaN
        initial_missing = data.isna().sum().sum()
        
        logger.debug(f"Missing values before processing infinite values: {initial_missing:,}")
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        final_missing = data.isna().sum().sum()
        inf_converted = final_missing - initial_missing
        
        logger.debug(f"Missing values after processing infinite values: {final_missing:,}")
        logger.debug(f"Infinite values converted to NaN: {inf_converted:,}")
        
        # Analyze missing values
        missing = data.isna().sum()
        missing_cols = missing[missing > 0]
        
        if len(missing_cols) > 0 and logger:
            mis_per = (missing / len(data)) * 100
            logger.debug("Columns with missing values:")
            for col in missing_cols.index:
                count = missing[col]
                percentage = mis_per[col]
                logger.debug(f"  {col}: {count:,} ({percentage:.2f}%)")
                
        if flag ==0:
            # Handle specific columns with median imputation
            flow_cols = ['Flow Bytes/s', 'Flow Packets/s']
            for col in flow_cols:
                if col in data.columns:
                    median_value = data[col].median()
                    missing_count = data[col].isna().sum()
                    
                    if missing_count > 0:
                        logger.debug(f"Filling {missing_count:,} missing values in '{col}' with median: {median_value:.2f}")
                        data.fillna({col: median_value}, inplace=True)
        
            # Verify no missing values remain in flow columns
            for col in flow_cols:
                    if col in data.columns:
                        remaining_missing = data[col].isna().sum()
                        logger.debug(f"Remaining missing values in '{col}': {remaining_missing}")

        # Dropping columns with only one unique value
        num_unique = data.nunique()
        one_variable = num_unique[num_unique == 1]
        not_one_variable = num_unique[num_unique > 1].index

        dropped_cols = one_variable.index
        data = data[not_one_variable]

        logger.debug(f"Dropping {len(dropped_cols)} columns with only one unique value:")
        for col in dropped_cols:
            logger.debug(f"  Dropped column: {col}")
        
        # Final dimensions
        final_rows, final_cols = data.shape
        total_removed = rows - final_rows
        retention_rate = (final_rows / rows * 100)
        
        logger.info("=" * 60)
        logger.info("Preprocessing completed successfully")
        logger.info(f"Final dimensions: {final_rows:,} rows x {final_cols} columns")
        logger.info(f"Total rows removed: {total_removed:,} ({(total_removed / rows * 100):.2f}%)")
        logger.info(f"data retention rate: {retention_rate:.2f}%")
        logger.info("=" * 60)
        
        return data
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError:
        error_msg = "The CSV file is empty"
        logger.error(error_msg)
        raise ValueError(error_msg)
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise