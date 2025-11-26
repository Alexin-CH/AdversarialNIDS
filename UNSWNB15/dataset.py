import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

root_dir = os.getcwd().split("AdversarialNIDS")[0] + "AdversarialNIDS"
sys.path.append(root_dir)

from scripts.logger import SimpleLogger

from UNSWNB15.preprocessing.download import download_prepare
from UNSWNB15.preprocessing.memory_optimization import optimize_memory_usage
from UNSWNB15.preprocessing.encoding import data_encoding
from UNSWNB15.preprocessing.scaling import scale
from UNSWNB15.preprocessing.spliting import split_data
from UNSWNB15.preprocessing.subset import subset_indices

from UNSWNB15.analysis.distribution import data_distribution
from UNSWNB15.analysis.mutual_info import mutual_info_classif
from UNSWNB15.analysis.pca import apply_pca
from UNSWNB15.analysis.features import FIRST_LEVEL_FEATURES, MODIFIABLE_FEATURES

class UNSWNB15():
    def __init__(self, dataset_size="small", logger=SimpleLogger()):
        """ Initialize the UNSWNB15 dataset class by downloading and preparing the dataset. """
        self.logger = logger
        self.data = download_prepare(logger=self.logger, dataset_size=dataset_size)
        self.multi_class = True
        self.FIRST_LEVEL_FEATURES = FIRST_LEVEL_FEATURES
        self.MODIFIABLE_FEATURES = MODIFIABLE_FEATURES

        self.categorical_cols = ["proto", "state", "service", "sport", "Destination Port"]
        for col in self.categorical_cols:
            self.data[col] = self.data[col].astype(str) # For Label Encoding compatibility

    def optimize_memory(self):
        """ Optimize memory usage of the dataset. """
        self.logger.info("Optimizing memory usage of the dataset...")
        self.data = optimize_memory_usage(self.data, logger=self.logger)
        return self

    def encode(self, attack_encoder="label"):
        """ Encode the dataset using data_encoding function. """
        self.logger.info("Encoding attack labels...")
        encoded = data_encoding(self.data, self.categorical_cols, attack_encoder=attack_encoder, logger=self.logger)
        self.features, self.is_attack, self.attack_classes = encoded

        for col in self.categorical_cols:
            self.features[col] = self.features[col].astype(float) # For Scaling compatibility
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
        return self

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
        
    def distribution(self):
        """ Display the distribution of attack classes in the dataset. """
        self.logger.info("Calculating data distribution...")

        if self.multi_class:
            data = self.attack_classes
        else:
            data = self.is_attack

        distribution = data_distribution(
            data=data,
            logger=self.logger
        )
        return distribution

    def mutual_info(self):
        """ Calculate mutual information for feature selection. """
        self.logger.info("Calculating mutual information for feature selection...")

        X = self.features.values.astype(float)

        if self.multi_class:
            y = self.attack_classes.values.astype(float)
        else:
            y = self.is_attack.values.astype(float)

        mi = mutual_info_classif(
            X=X,
            y=y,
            logger=self.logger
        )
        return mi

    def pca(self, n_components=2):
        """ Apply PCA to reduce dimensionality of the dataset. """
        self.logger.info(f"Applying PCA with {n_components} components...")

        data = self.features.values.astype(float)

        principal_components, explained_variance_ratio = apply_pca(
            data=data,
            n_components=n_components
        )
        self.logger.info(f"Explained variance ratio: {explained_variance_ratio.sum():.4f}")
        return principal_components, explained_variance_ratio
