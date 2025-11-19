# Module Classificateur K-Nearest Neighbors (KNN)

Ce module fournit des fonctions pour créer, entraîner, ajuster et évaluer des modèles K-Nearest Neighbors (KNN). Il inclut des utilitaires pour l'entraînement autonome, la sélection du k optimal, la recherche par grille et l'estimation de la mémoire utilisée.

## Aperçu des Fonctions

### train_knn(...)
Entraîne un KNeighborsClassifier autonome sans pipeline.
Effectue une validation croisée puis ajuste le modèle sur l'ensemble complet.
Ne réalise pas de mise à l'échelle ni de SMOTE.
Utile pour des expériences rapides lorsque le prétraitement est déjà fait.

### find_optimal_k(...)
Teste plusieurs valeurs de k via validation croisée pour trouver le meilleur nombre de voisins.
Retourne les k testés, les scores moyens, les écarts-types et le k optimal.

### tune_knn_hyperparameters(...)
Effectue une GridSearchCV sur plusieurs hyperparamètres (n_neighbors, weights, metric, p).
Retourne les meilleurs paramètres, le meilleur score et l'instance GridSearchCV ajustée.

### estimate_knn_memory(...)
Estime l'utilisation mémoire d'un modèle KNN selon le nombre d'échantillons et de variables.
KNN stocke tout l'ensemble d'entraînement, donc la mémoire peut être importante.
Retourne l'utilisation mémoire en Mo.

