import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

root_dir = os.getcwd().split("AdversarialNIDS")[0] + "AdversarialNIDS"
sys.path.append(root_dir)

from scripts.logger import SimpleLogger

def scale(data, scaler="standard", logger=SimpleLogger()):
    """ Scale the features of the dataset using a scaler. """
    available_scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler()
    }
    if scaler not in available_scalers:
        logger.error(f"Scaler '{scaler}' is not recognized. Available scalers: {list(available_scalers.keys())}")
        raise ValueError(f"Scaler '{scaler}' is not recognized.")

    scaler = available_scalers[scaler]
    scaled_features = scaler.fit_transform(data)

    logger.info(f"Features scaled using {scaler} scaler.")
    return pd.DataFrame(scaled_features, columns=data.columns)
