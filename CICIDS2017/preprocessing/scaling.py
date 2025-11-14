import os
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler

current_dir = os.getcwd()
sys.path.append(current_dir)

from scripts.logger import SimpleLogger

def scale(data, scaler=StandardScaler(), logger=SimpleLogger()):
    """ Scale the features of the dataset using a scaler. """
    features = data.drop(columns=['Attack Type'])
    scaled_features = scaler.fit_transform(features)

    logger.info(f"Features scaled using {scaler.__class__.__name__}")
    return scaled_features
