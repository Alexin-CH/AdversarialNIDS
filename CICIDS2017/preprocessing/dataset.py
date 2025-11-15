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

    def encode(self, attack_encoder="label"):
        """ Encode the dataset using data_encoding function. """
        self.logger.info("Encoding attack labels...")
        encoded = data_encoding(self.data, attack_encoder=attack_encoder, logger=self.logger)
        self.data, self.is_attack, self.attack_classes = encoded
        return self

    def optimize_memory(self):
        """ Optimize memory usage of the dataset. """
        self.logger.info("Optimizing memory usage of the dataset...")
        self.data = optimize_memory_usage(self.data, logger=self.logger)
        return self

    def scale(self, scaler="standard", logger=SimpleLogger()):
        """ Scale the dataset features using the provided scaler. """
        self.logger.info("Scaling dataset features...")
        self.scaled_features = scale(self.data, scaler=scaler, logger=self.logger)
        return self
        
