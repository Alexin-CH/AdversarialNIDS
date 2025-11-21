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
from CICIDS2017.preprocessing.subset import subset_indices
from CICIDS2017.preprocessing.download import data_distribution

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
        self.features, self.is_attack, self.attack_classes = encoded
        return self

    def scale(self, scaler="standard"):
        """ Scale the dataset features using the provided scaler. """
        self.logger.info("Scaling dataset features...")
        self.features = scale(self.features, scaler=scaler, logger=self.logger)
        return self

    def subset(self, size=None, multi_class=False):
        """ Undersample the dataset to the specified size. """
        self.logger.info(f"Subsetting dataset to size: {size}...")
        self.multi_class = multi_class

        if self.multi_class:
            data = self.attack_classes
        else:
            data = self.is_attack

        indices = subset_indices(
            data=data,
            size=size,
            logger=self.logger
        )
        self.features = self.features.iloc[indices].reset_index(drop=True)
        self.is_attack = self.is_attack.iloc[indices].reset_index(drop=True)
        self.attack_classes = self.attack_classes.iloc[indices].reset_index(drop=True)
        return self, self.multi_class

    def split(self, test_size=0.2, to_tensor=False, one_hot=False, apply_smote=False):
        """ Split the dataset into training and testing sets. """
        self.logger.info("Splitting dataset into training and testing sets...")

        X = self.features.values.astype(float)

        if self.multi_class:
            y = self.attack_classes.values.astype(float)
        else:
            y = self.is_attack.values.astype(float)

        data_split = split_data(
            X=X,
            y=y,
            test_size=test_size,
            to_tensor=to_tensor,
            apply_smote=apply_smote,
            one_hot=one_hot,
            logger=self.logger
        )
        return data_split
        
    def distribution(self, data):
        """ Display the distribution of attack classes in the dataset. """
        self.logger.info("Calculating data distribution...")

        distribution = data_distribution(
            data,
            logger=self.logger
        )
        return distribution