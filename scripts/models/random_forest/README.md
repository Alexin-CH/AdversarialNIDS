# Module Classificateur Random Forest

Ce module fournit des fonctions spécifiques à l'entraînement, l'ajustement et l'utilisation de Random Forest. Il inclut des utilitaires pour la sélection de variables, l'entraînement autonome avec validation croisée, la recherche par grille et le calcul du score OOB.

## Aperçu des Fonctions

### train_random_forest(...)
Entraîne un RandomForestClassifier autonome sans pipeline de prétraitement.
Effectue une validation croisée puis ajuste le modèle sur l'ensemble complet.
Aucune mise à l'échelle ni SMOTE appliquée sauf si fait manuellement avant.

### tune_rf_hyperparameters(...)
Effectue une GridSearchCV pour trouver les meilleurs hyperparamètres du RandomForestClassifier.
Retourne les meilleurs paramètres, le meilleur score et l'objet GridSearchCV.

### get_rf_oob_score(...)
Entraîne un RandomForestClassifier avec évaluation out-of-bag (OOB).
Retourne le modèle et son score OOB.