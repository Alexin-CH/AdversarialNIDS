# Analyse CICIDS2017

Ce dossier contient des scripts d'analyse pour le dataset CICIDS2017. Ces outils permettent d'explorer et de comprendre les données avant l'entraînement des modèles de détection d'intrusion.

## Modules

### 1. `distribution.py` - Distribution des données

**Fonction :** `data_distribution(data, logger=SimpleLogger())`

Affiche la distribution des données par type d'attaque.

**Paramètres :**
- `data` : DataFrame pandas ou série contenant les données avec une colonne 'Attack Type'
- `logger` : Logger pour afficher les informations (optionnel)

**Retour :**
- Distribution des types d'attaques sous forme de Series pandas

**Fonctionnalités :**
- Compte le nombre d'instances par type d'attaque
- Affiche les statistiques de distribution formatées
- Gestion d'erreurs robuste avec logging

**Exemple d'utilisation :**
```python
from CICIDS2017.analysis.distribution import data_distribution

distribution = data_distribution(df)
# Affiche : "Data Distribution by Attack Type:"
#           "  BENIGN: 2,273,097 instances"
#           "  DoS: 128,027 instances"
#           etc.
```

---

### 2. `features.py` - Gestion des features

Ce module contient deux fonctions principales pour la gestion des features du dataset.

#### `compute_features_batch(X, dico, eps=1e-6)`

Calcule des features dérivées par lot (batch) en utilisant PyTorch pour des performances optimales.

**Paramètres :**
- `X` : Tenseur PyTorch contenant les données
- `dico` : Dictionnaire de mapping des noms de features vers leurs indices
- `eps` : Valeur epsilon pour éviter les divisions par zéro (défaut: 1e-6)

**Features calculées :**
- **Fwd Packets/s** : Nombre de paquets forward par seconde
- **Bwd Packets/s** : Nombre de paquets backward par seconde
- **Avg Fwd Segment Size** : Taille moyenne des segments forward
- **Avg Bwd Segment Size** : Taille moyenne des segments backward
- **Sintpkt** : Temps inter-arrivée moyen des paquets source
- **Dintpkt** : Temps inter-arrivée moyen des paquets destination

**Sécurité :**
- Utilise `torch.clamp()` pour éviter les divisions par zéro
- Gestion des cas limites avec epsilon

#### `first_level_features(self)`

Retourne la liste des features de premier niveau du dataset.

**Categories de features :**
- **Identifiants réseau** : srcip, sport, dstip, dsport, proto, service
- **Durée** : dur
- **Paquets** : spkts, dpkts
- **Octets** : sbytes, dbytes
- **TTL** : sttl, dttl
- **Fenêtres** : swin, dwin
- **État** : state
- **TCP** : synack, ackdat, tcprtt
- **Pertes** : sloss, dloss
- **HTTP/FTP** : trans_depth, res_bdy_len, ct_ftp_cmd, ct_flw_http_mthd
- **Statistiques temporelles** : ct_srv_src, ct_srv_dst, ct_dst_ltm, ct_src_ltm, ct_src_dport_ltm, ct_dst_sport_ltm

---

### 3. `mutual_info.py` - Information Mutuelle

**Fonction :** `mutual_info_classif(X, y, logger=SimpleLogger())`

Calcule l'information mutuelle entre les features et les labels pour la sélection de features.

**Paramètres :**
- `X` : Matrice des features
- `y` : Vecteur des labels
- `logger` : Logger pour afficher les informations (optionnel)

**Retour :**
- Array numpy contenant les scores d'information mutuelle pour chaque feature

**Fonctionnalités :**
- Utilise `sklearn.feature_selection.mutual_info_classif`
- Détection automatique des features discrètes
- Calcul parallélisé avec 10 jobs
- Affichage détaillé des scores MI pour chaque feature

**Utilité :**
L'information mutuelle mesure la dépendance entre une feature et la variable cible. Plus le score est élevé, plus la feature est informative pour la classification.

---

### 4. `pca.py` - Analyse en Composantes Principales (PCA)

**Fonction :** `apply_pca(data, n_components=2)`

Applique une Analyse en Composantes Principales (PCA) pour la réduction de dimensionnalité.

**Paramètres :**
- `data` : Matrice des données à réduire
- `n_components` : Nombre de composantes principales à conserver (défaut: 2)

**Retour :**
- `principal_components` : Données transformées dans l'espace réduit
- `explained_variance_ratio` : Proportion de variance expliquée par chaque composante

**Utilité :**
- Visualisation des données en 2D/3D
- Réduction de dimensionnalité pour accélérer l'entraînement
- Comprendre la structure des données

**Exemple d'utilisation :**
```python
from CICIDS2017.analysis.pca import apply_pca

# Réduction en 2 dimensions pour visualisation
components, variance_ratio = apply_pca(X_scaled, n_components=2)
print(f"Variance expliquée : {sum(variance_ratio):.2%}")
```

---

## Workflow d'analyse recommandé

1. **Distribution des données** (`distribution.py`)
   - Vérifier l'équilibre des classes
   - Identifier les types d'attaques présents

2. **Analyse des features** (`features.py`)
   - Calculer les features dérivées
   - Comprendre les features de premier niveau

3. **Sélection de features** (`mutual_info.py`)
   - Identifier les features les plus informatives
   - Éliminer les features redondantes

4. **Visualisation** (`pca.py`)
   - Projeter les données en 2D/3D
   - Visualiser la séparabilité des classes
