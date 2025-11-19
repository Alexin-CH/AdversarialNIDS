import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

def split_data(X, y, test_size=0.2, toTensor=False, apply_smote=False, oneHot=False, logger=None):
    """ Splits the dataset into training and testing sets. """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if apply_smote:
        smote = SMOTE()
        X_train, y_train = smote.fit_resample(X_train, y_train)
        if logger:
            logger.debug("Applied SMOTE to balance the training set.")

    if logger:
        # Display num of elements per class
        unique, counts = np.unique(y_train, return_counts=True)
        logger.info("Class distribution after SMOTE:")
        for i in range(len(unique)):
            logger.info(f"  Class {unique[i]}: {counts[i]} samples")

    if oneHot:
        encoder = OneHotEncoder(sparse_output=False)
        y_train = encoder.fit_transform(y_train.reshape(-1, 1))
        y_test = encoder.transform(y_test.reshape(-1, 1))
        if logger:
            logger.debug(f"One-hot encoded classes: {encoder.categories_}")

    if toTensor:
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        if logger:
            logger.debug("Converted data to PyTorch tensors.")

    return X_train, X_test, y_train, y_test
