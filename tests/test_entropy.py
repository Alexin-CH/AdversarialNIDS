import os
import sys
from datetime import datetime

root_dir = os.getcwd().split("AdversarialNIDS")[0] + "AdversarialNIDS"
sys.path.append(root_dir)

from scripts.logger import LoggerManager
from CICIDS2017.preprocessing.dataset import CICIDS2017

lm = LoggerManager(log_dir=f"{root_dir}/logs", log_name="test_entropy")
lm.logger.info("Logger initialized")

dataset = CICIDS2017(logger=lm.logger).encode(attack_encoder="label").optimize_memory().scale(scaler="minmax")

##############################################
###############  Entropy Analysis  ###############
##############################################

from sklearn.feature_selection import mutual_info_classif

X = dataset.scaled_features
y = dataset.attack_classes

lm.logger.info("Calculating mutual information for feature selection")
mi = mutual_info_classif(X, y, discrete_features='auto', n_jobs=10)

# Display mutual information scores
for idx, score in enumerate(mi):
    lm.logger.info(f"Feature {dataset.data.columns[idx]} ({idx}): MI Score = {score}")

# This score indicates the dependency between each feature and the target variable (attack classes).
