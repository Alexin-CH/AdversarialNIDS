# Prétraitement du Dataset CICIDS2017

Ce répertoire contient l'ensemble des modules Python nécessaires au prétraitement du dataset CICIDS2017 pour l'entraînement de systèmes de détection d'intrusion réseau (NIDS).

## Vue d'ensemble du workflow

Le prétraitement du dataset CICIDS2017 s'effectue en plusieurs étapes séquentielles :

1. **Téléchargement et préparation** (`download.py`)
2. **Optimisation mémoire** (`memory_optimization.py`)
3. **Encodage des labels** (`encoding.py`)
4. **Normalisation des features** (`scaling.py`)
5. **Sous-échantillonnage** (`subset.py`)
6. **Division train/test** (`spliting.py`)

La classe principale `CICIDS2017` dans `dataset.py` orchestre l'ensemble de ce workflow.

---

## Modules

### 1. `download.py` - Téléchargement et préparation

#### Fonction `download_prepare(logger=SimpleLogger())`

Télécharge le dataset CICIDS2017 depuis Kaggle et effectue le nettoyage initial.

**Étapes de prétraitement :**
1. Téléchargement du dataset via `kagglehub`
2. Nettoyage des noms de colonnes (suppression des espaces)
3. Suppression des doublons
4. Suppression des lignes avec valeurs manquantes
5. Gestion des valeurs infinies (conversion en NaN)
6. Imputation par la médiane pour les colonnes 'Flow Bytes/s' et 'Flow Packets/s'
7. Suppression des colonnes à variance nulle (une seule valeur unique)
8. Mapping des labels vers les types d'attaque

**Mapping des attaques :**
```python
attack_map = {
    'BENIGN': 'BENIGN',
    'DDoS': 'DDoS',
    'DoS Hulk': 'DoS',
    'DoS GoldenEye': 'DoS',
    'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS',
    'PortScan': 'Port Scan',
    'FTP-Patator': 'Brute Force',
    'SSH-Patator': 'Brute Force',
    'Bot': 'Bot',
    'Web Attack Brute Force': 'Web Attack',
    'Web Attack XSS': 'Web Attack',
    'Web Attack Sql Injection': 'Web Attack',
    'Infiltration': 'Infiltration',
    'Heartbleed': 'Heartbleed'
}
```

**Retourne :** `pd.DataFrame` - Dataset nettoyé avec colonne 'Attack Type'

---

### 2. `memory_optimization.py` - Optimisation mémoire

#### Fonction `optimize_memory_usage(data, logger=None)`

Réduit l'empreinte mémoire du DataFrame en convertissant les types numériques vers des types de précision inférieure.

**Optimisations :**
- `float64` → `float32` (si les valeurs sont dans la plage supportée)
- `int64` → `int32` (si les valeurs sont dans la plage supportée)

**Paramètres :**
- `data` : DataFrame à optimiser
- `logger` : Logger pour afficher les statistiques de réduction mémoire

**Retourne :** `pd.DataFrame` - DataFrame optimisé

**Gain typique :** ~50% de réduction de la consommation mémoire

---

### 3. `encoding.py` - Encodage des labels

#### Fonction `data_encoding(data, attack_encoder="label", logger=SimpleLogger())`

Encode les labels d'attaque en valeurs numériques.

**Encodeurs disponibles :**
- `"label"` : LabelEncoder - Encode chaque classe par un entier unique
- `"onehot"` : OneHotEncoder - Encode en vecteurs binaires

**Paramètres :**
- `data` : DataFrame contenant la colonne 'Attack Type'
- `attack_encoder` : Type d'encodeur ("label" ou "onehot")
- `logger` : Logger pour le suivi

**Retourne :** Tuple `(data, is_attack, attack_classes)`
- `data` : DataFrame original
- `is_attack` : Array binaire (0=BENIGN, 1=ATTAQUE)
- `attack_classes` : Array des classes d'attaque encodées

---

### 4. `scaling.py` - Normalisation

#### Fonction `scale(data, scaler="standard", logger=SimpleLogger())`

Normalise les features numériques du dataset (excluant 'Attack Type').

**Scalers disponibles :**
- `"standard"` : StandardScaler - Normalisation Z-score (moyenne=0, écart-type=1)
- `"minmax"` : MinMaxScaler - Normalisation min-max (plage [0, 1])

**Paramètres :**
- `data` : DataFrame contenant les features et 'Attack Type'
- `scaler` : Type de normalisation
- `logger` : Logger pour le suivi

**Retourne :** `np.ndarray` - Features normalisées

---

### 5. `subset.py` - Sous-échantillonnage

#### Fonction `subset_indices(data, size=None, logger=None)`

Effectue un sous-échantillonnage équilibré par classe.

**Stratégie :**
- Calcule `max_size = size // nombre_de_classes`
- Pour chaque classe, sélectionne aléatoirement jusqu'à `max_size` échantillons
- Équilibre automatiquement les classes

**Paramètres :**
- `data` : Array des labels
- `size` : Taille totale cible du dataset (si None, utilise la taille maximale)
- `logger` : Logger pour afficher la distribution

**Retourne :** `list` - Liste des indices sélectionnés

---

### 6. `spliting.py` - Division train/test

#### Fonction `split_data(X, y, test_size=0.2, to_tensor=False, apply_smote=False, one_hot=False, logger=None)`

Divise le dataset en ensembles d'entraînement et de test avec options avancées.

**Fonctionnalités :**
- Division stratifiée train/test
- Application optionnelle de SMOTE pour équilibrer les classes minoritaires
- Encodage one-hot optionnel des labels
- Conversion optionnelle en tenseurs PyTorch

**Paramètres :**
- `X` : Features (array ou DataFrame)
- `y` : Labels (array)
- `test_size` : Proportion de l'ensemble de test (défaut: 0.2)
- `to_tensor` : Convertit en tenseurs PyTorch si True
- `apply_smote` : Applique SMOTE sur l'ensemble d'entraînement si True
- `one_hot` : Encode les labels en one-hot si True
- `logger` : Logger pour afficher les statistiques

**Retourne :** Tuple `(X_train, X_test, y_train, y_test)`

**Note sur SMOTE :** Utilise la stratégie 'not majority' pour sur-échantillonner uniquement les classes minoritaires.

---

## Workflow complet - Exemple

```python
from CICIDS2017.preprocessing.dataset import CICIDS2017
from scripts.logger import SimpleLogger

# Initialisation
logger = SimpleLogger()
dataset = CICIDS2017(logger=logger)
# Optimisation mémoire
dataset = dataset.optimize_memory()
# Encodage des labels
dataset = dataset.encode(attack_encoder="label")
# Normalisation des features
dataset = dataset.scale(scaler="standard")
# Sous-échantillonnage
dataset = dataset.subset(size=100000, multi_class=False)

# Division et préparation finale
X_train, X_test, y_train, y_test = dataset.split(
    test_size=0.2,
    to_tensor=True,
    apply_smote=True,
    one_hot=False
)
```

---

## Notes importantes

1. **Ordre des opérations** : L'ordre des méthodes est important. Le workflow recommandé est :
   `optimize_memory()` → `encode()` → `scale()` → `subset()` → `split()`

2. **Gestion mémoire** : Le dataset CICIDS2017 complet peut consommer beaucoup de mémoire. Utilisez `optimize_memory()` dès le début.

3. **SMOTE** : N'appliquez SMOTE que si vous avez un déséquilibre significatif entre les classes.

4. **Multi-class vs Binary** : Choisissez `multi_class=True` pour la classification multi-classe (types d'attaque), ou `False` pour la détection binaire (attaque/bénin).

5. **Reproductibilité** : Pour des résultats reproductibles, définissez une seed aléatoire avant d'utiliser `subset()` et `split()`.