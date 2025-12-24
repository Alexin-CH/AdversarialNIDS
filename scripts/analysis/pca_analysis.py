import os
import sys
from datetime import datetime

root_dir = os.getcwd().split("AdversarialNIDS")[0] + "AdversarialNIDS"
sys.path.append(root_dir)

from scripts.logger import LoggerManager
from CICIDS2017.dataset import CICIDS2017

lm = LoggerManager(
    root_dir=root_dir,
    log_name="PCA_Analysis"
)
lm.logger.info("Logger initialized")

dataset = CICIDS2017(logger=lm.logger).encode().optimize_memory()

##############################################
###############  PCA Analysis  ###############
##############################################

print("=" * 50)

from sklearn.decomposition import PCA

pca_componants = []
for n_components in range(1, len(dataset.data.columns), 1):
    pca, exp_var_ratio = dataset.pca(n_components=n_components)
    pca_componants.append(exp_var_ratio)
    lm.logger.info(f"PCA with {n_components} components")
    lm.logger.info(f"Explained variance ratio: {exp_var_ratio.sum():.4f}")
    lm.logger.info("-" * 50)

import matplotlib.pyplot as plt

explained_variances = [v.sum() for v in pca_componants]
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variances) + 1), explained_variances, marker='o')
plt.title('Explained Variance vs Number of PCA Components')
plt.xlabel('Number of PCA Components')
plt.ylabel('Explained Variance Ratio')
plt.grid()
os.makedirs(f"{root_dir}/results/PCA_Analysis", exist_ok=True)
plt.savefig(f"{root_dir}/results/PCA_Analysis/explained_variance.png")
plt.close()
