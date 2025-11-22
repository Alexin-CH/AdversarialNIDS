import os
import sys
import pandas as pd

root_dir = os.getcwd().split("AdversarialNIDS")[0] + "AdversarialNIDS"
sys.path.append(root_dir)

from scripts.logger import SimpleLogger

def data_distribution(data, logger=SimpleLogger()):
    """ Display the distribution of data by attack type. """
    try:
        if type(data) is pd.DataFrame:
            distribution = data['Attack Type'].value_counts()
        else:
            distribution = pd.Series(data).value_counts()
        logger.info("Data Distribution by Attack Type:")
        for attack_type, count in distribution.items():
            logger.info(f"  {attack_type}: {count:,} instances")
        
        return distribution
    
    except KeyError as e:
        logger.error(f"Key error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error calculating data distribution: {e}")
        raise
