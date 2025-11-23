# Scripts d'Analyse

Ce dossier contient des scripts Python pour l'analyse et l'évaluation des modèles de classification dans le cadre du projet AdversarialNIDS.

## Modules

### 1. `classification_report.py`

Fournit des fonctions pour visualiser les rapports de classification sous forme de tableaux formatés.

#### Fonctions principales :

**`plot_classification_report(report_dict, ax, title)`**
- Affiche un rapport de classification comme un tableau formaté avec matplotlib
- **Paramètres :**
  - `report_dict` : Dictionnaire de rapport de classification provenant de sklearn
  - `ax` : Axe matplotlib sur lequel tracer
  - `title` : Titre du tableau
- **Fonctionnalités :**
  - Affiche les métriques : precision, recall, f1-score, support
  - Inclut les moyennes macro et weighted
  - Mise en forme avec en-tête coloré et séparateurs
  - Ajustement automatique de la taille des polices

---

### 2. `model_analysis.py`

Script principal pour effectuer une analyse complète des modèles de classification.

#### Fonctions principales :

**`perform_model_analysis(model, X_test, y_test, dir=root_dir, logger=SimpleLogger(), model_name="Model", class_names=None, plot=True, save_fig=True, device=None)`**

Effectue une analyse complète de classification avec matrice de confusion et visualisation du rapport.

- **Paramètres :**
  - `model` : Modèle de classification entraîné (PyTorch nn.Module ou modèle sklearn)
  - `X_test` : Caractéristiques de test (numpy array, pandas DataFrame, ou torch Tensor)
  - `y_test` : Étiquettes vraies (numpy array, pandas Series, ou torch Tensor)
  - `dir` : Répertoire pour sauvegarder les figures (défaut: root_dir)
  - `logger` : Instance de logger pour enregistrer les résultats (défaut: SimpleLogger())
  - `model_name` : Nom du modèle pour les titres et logs (défaut: "Model")
  - `plot` : Afficher les graphiques (défaut: True)
  - `save_fig` : Sauvegarder les figures (défaut: True)
  - `device` : Device pour les modèles PyTorch ('cuda' ou 'cpu', défaut: auto-détection)

- **Retourne :**
  - `cm` : Matrice de confusion
  - `report_dict` : Dictionnaire du rapport de classification

- **Fonctionnalités :**
  - Compatible avec les modèles PyTorch et scikit-learn
  - Détection automatique du type de modèle
  - Génération de matrice de confusion avec heatmap
  - Tableau détaillé du rapport de classification
  - Enregistrement des résultats dans les logs
  - Sauvegarde automatique des visualisations

**Dépendances :**
- torch, numpy, matplotlib, seaborn
- sklearn.metrics (confusion_matrix, classification_report)
- Scripts internes : pytorch_prediction, classification_report, logger

---

### 3. `pytorch_prediction.py`

Utilitaire pour obtenir des prédictions à partir de modèles PyTorch.

#### Fonctions principales :

**`get_pytorch_predictions(model, X_test, y_test, device, batch_size=32)`**

Obtient les prédictions d'un modèle PyTorch avec traitement par batch.

- **Paramètres :**
  - `model` : Modèle PyTorch
  - `X_test` : Caractéristiques de test
  - `y_test` : Étiquettes vraies
  - `device` : Device d'exécution ('cuda' ou 'cpu')
  - `batch_size` : Taille de batch pour l'inférence (défaut: 32)

- **Retourne :**
  - `tuple` : (predictions, true_labels) en tant que numpy arrays

- **Fonctionnalités :**
  - Conversion automatique des données (numpy, pandas, torch tensors)
  - Traitement par batch avec DataLoader
  - Gestion des étiquettes one-hot encodées
  - Mode évaluation automatique
  - Inférence sans calcul de gradients (torch.no_grad())

## Utilisation

### Exemple d'analyse complète d'un modèle

```python
from scripts.analysis.model_analysis import perform_model_analysis
from scripts.logger import SimpleLogger

# Initialiser le logger
logger = SimpleLogger()

# Analyser un modèle PyTorch
cm, report = perform_model_analysis(
    model=my_pytorch_model,
    X_test=X_test_data,
    y_test=y_test_data,
    logger=logger,
    model_name="My Deep Learning Model",
    device="cuda"
)

# Analyser un modèle scikit-learn
cm, report = perform_model_analysis(
    model=my_sklearn_model,
    X_test=X_test_data,
    y_test=y_test_data,
    logger=logger,
    model_name="Random Forest Classifier",
)
```

## Sorties

L'analyse génère :
1. **Logs texte** : Rapport de classification détaillé dans les logs
2. **Visualisations** :
   - Matrice de confusion (heatmap colorée)
   - Tableau du rapport de classification (precision, recall, f1-score, support)
3. **Fichiers PNG** : Sauvegardés automatiquement avec le nom `{model_name}_model_analysis.png`

## Compatibilité

- **Frameworks supportés :** PyTorch, scikit-learn
- **Formats de données :** numpy arrays, pandas DataFrames/Series, torch Tensors
- **Devices :** CPU et GPU (CUDA)
