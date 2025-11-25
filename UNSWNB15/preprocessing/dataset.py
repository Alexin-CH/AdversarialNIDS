import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

current_dir = os.getcwd()
sys.path.append(current_dir)

from scripts.logger import SimpleLogger

from UNSWNB15.preprocessing.download import download_prepare
from UNSWNB15.preprocessing.memory_optimization import optimize_memory_usage
from UNSWNB15.preprocessing.encoding import data_encoding
from UNSWNB15.preprocessing.scaling import scale
from UNSWNB15.preprocessing.spliting import split_data
from UNSWNB15.preprocessing.subset import subset_indices

class UNSWNB15():
    def __init__(self, dataset_size="small", logger=SimpleLogger()):
        """ Initialize the UNSWNB15 dataset class by downloading and preparing the dataset. """
        self.logger = logger
        self.data = download_prepare(logger=self.logger, dataset_size=dataset_size)
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
        self.data, self.is_attack, self.attack_classes = encoded

        for col in self.categorical_cols:
            self.data[col] = self.data[col].astype(float) # For Scaling compatibility
        return self

    def scale(self, scaler="standard"):
        """ Scale the dataset features using the provided scaler. """
        self.logger.info("Scaling dataset features...")
        self.scaled_features = scale(self.data, scaler=scaler, logger=self.logger)
        return self
    
    def split(self, test_size=0.2, to_tensor=False, one_hot=False, apply_smote=False):
        """ Split the dataset into training and testing sets. """
        self.logger.info("Splitting dataset into training and testing sets...")

        data_split = split_data(
            X=self.scaled_features,
            y=self.attack_classes if self.multi_class else self.is_attack,
            test_size=test_size,
            to_tensor=to_tensor,
            one_hot=one_hot,
            apply_smote=apply_smote,
            logger=self.logger
        )
        return data_split

    def subset(self, size=None, multi_class=False):
        """ Undersample the dataset to the specified size. """
        self.logger.info(f"Subsetting dataset to size: {size}...")
        self.multi_class = multi_class

        indices = subset_indices(
            self.attack_classes if self.multi_class else self.is_attack,
            size=size,
            logger=self.logger
        )
        self.data = self.data.iloc[indices].reset_index(drop=True)
        self.scaled_features = self.scaled_features[indices]
        self.is_attack = self.is_attack[indices]
        self.attack_classes = self.attack_classes[indices]
        return self
    def first_level_features(self): 
        first_level = [
            "srcip",
            "sport",
            "dstip",
            "dsport",
            "proto",
            "service",

            "dur",

            "spkts",
            "dpkts",
            "sbytes",
            "dbytes",

            "sttl",
            "dttl",

            "swin",
            "dwin",

            "state",

            "synack",
            "ackdat",
            "tcprtt",

            "sloss",
            "dloss",

            "trans_depth",
            "res_bdy_len",
            "ct_ftp_cmd",
            "ct_flw_http_mthd",

            "ct_srv_src",
            "ct_srv_dst",
            "ct_dst_ltm",
            "ct_src_ltm",
            "ct_src_dport_ltm",
            "ct_dst_sport_ltm",
        ]
        return first_level