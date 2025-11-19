import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

def split_data(X, y, test_size=0.2, to_tensor=False, apply_smote=False, one_hot=False, logger=None):
    """ Splits the dataset into training and testing sets. """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    unique, counts = np.unique(y_train, return_counts=True)

    if logger:
        # Display num of elements per class
        logger.info("Class distribution before SMOTE:")
        for i in range(len(unique)):
            logger.info(f"  Class {unique[i]}: {counts[i]} samples")

    if apply_smote:
        smote = SMOTE(sampling_strategy='not majority')
        X_train, y_train = smote.fit_resample(X_train, y_train)
        if logger:
            logger.info("Applied SMOTE to balance the training set.")

    unique, counts = np.unique(y_train, return_counts=True)

    if logger:
        # Display num of elements per class
        logger.info("Class distribution after SMOTE:")
        for i in range(len(unique)):
            logger.info(f"  Class {unique[i]}: {counts[i]} samples")

    if one_hot:
        encoder = OneHotEncoder(sparse_output=False)
        y_train = encoder.fit_transform(y_train.reshape(-1, 1))
        y_test = encoder.transform(y_test.reshape(-1, 1))
        if logger:
            logger.debug(f"One-hot encoded classes: {encoder.categories_}")

    if to_tensor:
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        if logger:
            logger.debug("Converted data to PyTorch tensors.")

    return X_train, X_test, y_train, y_test
