import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

current_dir = os.getcwd()
sys.path.append(current_dir)

from scripts.logger import SimpleLogger

from CICIDS2017.preprocessing.download import download_prepare
from CICIDS2017.preprocessing.encoding import data_encoding
from CICIDS2017.preprocessing.scaling import scale
from CICIDS2017.preprocessing.memory_optimization import optimize_memory_usage

class CICIDS2017:
    def __init__(self, logger=SimpleLogger()):
        """ Initialize the CICIDS2017 dataset class by downloading and preparing the dataset. """
        self.logger = logger
        self.data = download_prepare(logger=self.logger)

    def encode(self):
        """ Encode the dataset using data_encoding function. """
        self.data = data_encoding(self.data, logger=self.logger)
        return self

    def optimize_memory(self):
        """ Optimize memory usage of the dataset. """
        self.data = optimize_memory_usage(self.data, logger=self.logger)
        return self

    def scale(self, scaler=StandardScaler(), keep_unscaled=True, logger=SimpleLogger()):
        """ Scale the dataset features using the provided scaler. """
        scaled_features = scale(self.data, scaler=scaler, logger=logger)
        if keep_unscaled:
            self.scaled_features = scaled_features
            self.logger.info("Scaled features are stored in 'scaled_features' attribute.")
        else:
            self.data = scaled_features
        return self
