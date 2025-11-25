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

#### 1\. Identification & Volume

| Feature | Description | Min / Max | Incrément | Corrélations Principales |
| :--- | :--- | :--- | :--- | :--- |
| **Destination Port** | Port cible (ex: 80 pour HTTP). | 0 - 65535 | 1 | Faible corrélation avec le reste. |
| **Flow Duration** | Durée totale de la connexion (s). | 0.0 - Variable | Continu | **Forte** avec `Flow IAT Sum`. Inverse avec `Flow Rate`. |
| **Total Fwd Packets** | Nombre de paquets envoyés. | 1 - Variable | 1 | **Très Forte** avec `Fwd Bytes`, `Subflow Fwd Pkts`. |
| **Total Bwd Packets** | Nombre de paquets reçus. | 0 - Variable | 1 | **Très Forte** avec `Bwd Bytes`, `Subflow Bwd Pkts`. |
| **Total Length Fwd/Bwd** | Volume total de données (Bytes). | 0 - Variable | 1 | **Très Forte** avec `Total Packets` correspondant. |

#### 2\. Tailles de Paquets (Packet Dynamics)

| Feature | Description | Min / Max | Corrélations Principales |
| :--- | :--- | :--- | :--- |
| **Fwd/Bwd Pkt Len Max** | Taille du plus gros paquet observé. | 0 - 65535 | Souvent corrélé avec `Packet Length Max`. |
| **Fwd/Bwd Pkt Len Min** | Taille du plus petit paquet (souvent header seul). | 0 - 1500 | Souvent constante (ex: 54 ou 66 bytes). |
| **Fwd/Bwd Pkt Len Mean** | Taille moyenne des paquets. | 0.0 - 1500 | Corrélé avec `Avg Segment Size`. |
| **Fwd/Bwd Pkt Len Std** | Variation de la taille (Jitter de taille). | 0.0 - Variable | Indicateur fort de payload complexe (ex: vidéo vs texte). |
| **Min/Max Pkt Len** | Bornes absolues sur tout le flux. | 0 - 65535 | Redondant avec `Fwd/Bwd Max`. |

#### 3\. Temps Inter-Arrivée (IAT - Inter-Arrival Time)

*L'IAT est le temps d'attente entre deux paquets consécutifs.*

| Feature | Description | Min / Max | Corrélations Principales |
| :--- | :--- | :--- | :--- |
| **Flow IAT Mean** | Temps moyen entre deux paquets. | 0.0 - Durée Flux | **Inverse** avec `Flow Packets/s` (Plus ça va vite, moins on attend). |
| **Flow IAT Std** | Régularité du flux (Jitter). | 0.0 - Variable | Forte pour les attaques DoS (très régulier = Std faible). |
| **Flow IAT Max** | Plus longue pause dans le flux. | 0.0 - Durée Flux | Souvent corrélé avec `Idle Max`. |
| **Fwd/Bwd IAT Total** | Durée totale active dans un sens. | 0.0 - Variable | Corrélé avec `Flow Duration`. |

#### 4\. Débit et Activité

| Feature | Description | Min / Max | Corrélations Principales |
| :--- | :--- | :--- | :--- |
| **Flow Bytes/s** | Vitesse de transfert (Octets). | 0.0 - Vitesse Lien | **Forte** avec `Flow Packets/s`. |
| **Flow Packets/s** | Vitesse de transfert (Paquets). | 0.0 - Variable | **Inverse** avec `Flow IAT Mean`. |
| **Active Mean/Max** | Temps d'activité avant une pause. | 0.0 - Variable | Corrélé avec `Flow Duration` sur les flux longs. |
| **Idle Mean/Max** | Temps de pause (Inactivité). | 0.0 - Variable | **Forte** avec `Flow IAT Max`. |

#### 5\. Indicateurs de Protocole (TCP Flags)

*Les drapeaux sont des compteurs (combien de fois ce drapeau a été vu).*

| Feature | Description | Min / Max | Corrélations Principales |
| :--- | :--- | :--- | :--- |
| **SYN Flag Count** | Début de connexion. | 0 - N | Élevé dans les attaques **DoS** et **Scan**. |
| **FIN Flag Count** | Fin de connexion propre. | 0 - N | Corrélé avec la fin normale des flux TCP. |
| **RST Flag Count** | Réinitialisation (Erreur/Attaque). | 0 - N | Indicateur d'attaques brutales ou scanners. |
| **ACK Flag Count** | Accusé de réception. | 0 - N | Très corrélé avec `Total Packets` (TCP normal). |
| **PSH Flag Count** | Pousser les données (Push). | 0 - N | Corrélé avec les transferts de données interactifs. |

#### 6\. Subflows et Fenêtres (Avancé)

| Feature | Description | Min / Max | Corrélations Principales |
| :--- | :--- | :--- | :--- |
| **Subflow Fwd/Bwd** | Caractéristiques de sous-flux agrégés. | 0 - Variable | **100% corrélé** (doublon) avec `Total Packets/Bytes` dans ce dataset spécifique. |
| **Init\_Win\_bytes** | Taille fenêtre TCP initiale. | 0 - 65535 | Dépend de l'OS (Windows/Linux ont des valeurs différentes). |
| **act\_data\_pkt\_fwd** | Paquets contenant vraiment de la donnée. | 0 - Variable | Corrélé avec `Total Fwd Packets` mais exclut les `ACK` vides. |
