import os
import sys
from datetime import datetime

current_dir = os.getcwd()
sys.path.append(current_dir)

from scripts.logger import LoggerManager
from CICIDS2017.preprocessing.dataset import CICIDS2017

lm = LoggerManager(log_dir=f"{current_dir}/logs", log_name="test_pca")
lm.logger.info("Logger initialized")

dataset = CICIDS2017(logger=lm.logger).encode().scale(scaler="minmax").optimize_memory()

##############################################
###############  PCA Analysis  ###############
##############################################

from sklearn.decomposition import PCA

pca_componants = []
for n_components in range(1, len(dataset.data.columns), 2):
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(dataset.scaled_features)
    pca_componants.append((n_components, pca, pca_features))
    lm.logger.info(f"PCA with {n_components} components")
    lm.logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    lm.logger.info(f"Components shape: {pca_features.shape}")
    lm.logger.info("-" * 50)

import matplotlib.pyplot as plt

explained_variances = [pca.explained_variance_ratio_.sum() for _, pca, _ in pca_componants]
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variances) + 1), explained_variances, marker='o')
plt.title('Explained Variance vs Number of PCA Components')
plt.xlabel('Number of PCA Components')
plt.ylabel('Explained Variance Ratio')
plt.grid()
plt.savefig(f"{current_dir}/logs/pca_explained_variance.png")
plt.close()
