# Dataset CICIDS2017 - Documentation

## Modules

Les fonctions relatives au prépraitement du dataset CICIDS2017 sont organisées dans le sous-répertoire `preprocessing/`. 

Les fonctions relatives aux analyses exploratoires et visualisations sont organisées dans le sous-répertoire `analysis/`.

### 1. `dataset.py` - Classe principale

#### Classe `CICIDS2017`

Classe principale qui encapsule l'ensemble du workflow de prétraitement.

**Constructeur :**
```python
CICIDS2017(dataset_size=None, logger=SimpleLogger())
```
- Initialise le dataset en téléchargeant et préparant les données
- `dataset_size` : Taille du dataset (optionnel)
- `logger` : Instance de logger pour le suivi des opérations

**Méthodes :**

- **`optimize_memory()`** : Optimise l'utilisation de la mémoire du dataset
  - Retourne : `self` (chaînage de méthodes)

- **`encode(attack_encoder="label")`** : Encode les labels d'attaque
  - `attack_encoder` : Type d'encodeur ("label" ou "onehot")
  - Retourne : `self` (chaînage de méthodes)
  - Crée les attributs : `self.is_attack`, `self.attack_classes`

- **`scale(scaler="standard")`** : Normalise les features du dataset
  - `scaler` : Type de normalisation ("standard" ou "minmax")
  - Retourne : `self` (chaînage de méthodes)
  - Crée l'attribut : `self.scaled_features`

- **`subset(size=None, multi_class=False)`** : Sous-échantillonne le dataset
  - `size` : Taille cible du dataset
  - `multi_class` : Si True, utilise les classes d'attaque ; sinon binaire (attaque/bénin)
  - Retourne : `self` (chaînage de méthodes)

- **`split(test_size=0.2, to_tensor=False, one_hot=False, apply_smote=False)`** : Divise en ensembles train/test
  - `test_size` : Proportion des données de test (défaut: 0.2)
  - `to_tensor` : Convertit en tenseurs PyTorch si True
  - `one_hot` : Applique l'encodage one-hot aux labels si True
  - `apply_smote` : Applique SMOTE pour équilibrer les classes si True
  - Retourne : `(X_train, X_test, y_train, y_test)`

- **`distribution()`** : Affiche la distribution des classes dans le dataset
  - Retourne : `distribution`

- **`mutual_info()`** : Calcule l'information mutuelle entre les features et les labels
  - Retourne : `mi`

- **`pca()`** : Applique l'Analyse en Composantes Principales (PCA) pour la réduction de dimensionnalité
  - Retourne : `pca`

**Exemple d'utilisation :**
```python
dataset = CICIDS2017().optimize_memory().encode().scale().subset(size=100000, multi_class=False)
X_train, X_test, y_train, y_test = dataset.split(test_size=0.2, apply_smote=True)
```

**Colonnes du dataset :**
```
Index(['Destination Port', 'Flow Duration', 'Total Fwd Packets',
       'Total Backward Packets', 'Total Length of Fwd Packets',
       'Total Length of Bwd Packets', 'Fwd Packet Length Max',
       'Fwd Packet Length Min', 'Fwd Packet Length Mean',
       'Fwd Packet Length Std', 'Bwd Packet Length Max',
       'Bwd Packet Length Min', 'Bwd Packet Length Mean',
       'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
       'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
       'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max',
       'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
       'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Fwd URG Flags',
       'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s',
       'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length',
       'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
       'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
       'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count',
       'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size',
       'Avg Bwd Segment Size', 'Fwd Header Length.1', 'Subflow Fwd Packets',
       'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
       'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd',
       'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max',
       'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'],
      dtype='object')
```