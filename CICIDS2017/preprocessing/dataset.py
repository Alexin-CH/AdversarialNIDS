import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

root_dir = os.getcwd().split("AdversarialNIDS")[0] + "AdversarialNIDS"
sys.path.append(root_dir)

from scripts.logger import SimpleLogger

from CICIDS2017.preprocessing.download import download_prepare
from CICIDS2017.preprocessing.memory_optimization import optimize_memory_usage
from CICIDS2017.preprocessing.encoding import data_encoding
from CICIDS2017.preprocessing.scaling import scale
from CICIDS2017.preprocessing.spliting import split_data

class CICIDS2017:
    def __init__(self, dataset_size=None, logger=SimpleLogger()):
        """ Initialize the CICIDS2017 dataset class by downloading and preparing the dataset. """
        self.logger = logger
        self.data = download_prepare(logger=self.logger)

    def optimize_memory(self):
        """ Optimize memory usage of the dataset. """
        self.logger.info("Optimizing memory usage of the dataset...")
        self.data = optimize_memory_usage(self.data, logger=self.logger)
        return self

    def encode(self, attack_encoder="label"):
        """ Encode the dataset using data_encoding function. """
        self.logger.info("Encoding attack labels...")
        encoded = data_encoding(self.data, attack_encoder=attack_encoder, logger=self.logger)
        self.data, self.is_attack, self.attack_classes = encoded
        return self


    def scale(self, scaler="standard"):
        """ Scale the dataset features using the provided scaler. """
        self.logger.info("Scaling dataset features...")
        self.scaled_features = scale(self.data, scaler=scaler, logger=self.logger)
        return self
        
    def split(self, multiclass=False, test_size=0.2, toTensor=False, oneHot=False, apply_smote=False):
        """ Split the dataset into training and testing sets. """
        self.logger.info("Splitting dataset into training and testing sets...")

        data_split = split_data(
            X=self.scaled_features,
            y=self.attack_classes if multiclass else self.is_attack,
            test_size=test_size,
            toTensor=toTensor,
            apply_smote=apply_smote,
            oneHot=oneHot,
            logger=self.logger
        )
        return data_split