import os
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler

current_dir = os.getcwd()
sys.path.append(current_dir)

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
    features = data.drop(columns=['Attack Type'])
    scaled_features = scaler.fit_transform(features)

    logger.info(f"Features scaled using {scaler} scaler.")
    return scaled_features
