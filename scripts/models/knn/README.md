# Module Classificateur K-Nearest Neighbors (KNN)

Ce module fournit des fonctions pour créer, entraîner, ajuster et évaluer des modèles K-Nearest Neighbors (KNN). Il inclut des utilitaires pour l'entraînement autonome, la sélection du k optimal, la recherche par grille.

## Aperçu des Fonctions

### train_knn(...)
Entraîne un KNeighborsClassifier
Effectue une validation croisée puis ajuste le modèle sur l'ensemble complet.

### find_optimal_k(...)
Teste plusieurs valeurs de k via validation croisée pour trouver le meilleur nombre de voisins.
Retourne les k testés, les scores moyens, les écarts-types et le k optimal.

### tune_knn_hyperparameters(...)
Effectue une GridSearchCV sur plusieurs hyperparamètres (n_neighbors, weights, metric, p).
Retourne les meilleurs paramètres, le meilleur score et l'instance GridSearchCV ajustée.


