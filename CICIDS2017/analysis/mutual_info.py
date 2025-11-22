import os
import sys
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

root_dir = os.getcwd().split("AdversarialNIDS")[0] + "AdversarialNIDS"
sys.path.append(root_dir)

from scripts.logger import SimpleLogger

def mutual_info_classif(X, y, logger=SimpleLogger()):
    """ Calculate mutual information for feature selection. """
    logger.info("Calculating mutual information for feature selection")
    mi = mutual_info_classif(X, y, discrete_features='auto', n_jobs=10)

    # Display mutual information scores
    for idx, score in enumerate(mi):
        logger.info(f"Feature {dataset.data.columns[idx]} ({idx}): MI Score = {score}")
    
    return mi
    