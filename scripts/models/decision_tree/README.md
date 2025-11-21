# Module Classificateur Arbre de Décision

Ce module fournit un ensemble de fonctions pour créer, entraîner, ajuster et analyser des modèles d'Arbre de Décision.

## Aperçu des Fonctions

### train_decision_tree(...)

Entraîne un DecisionTreeClassifier.
Effectue une validation croisée puis ajuste le modèle sur l'ensemble d'entraînement complet.

### analyze_tree_complexity(model, ...)

Analyse plusieurs métriques de complexité d'un arbre de décision entraîné :

* nombre de nœuds
* nombre de feuilles
* profondeur maximale
* nombre de variables effectivement utilisées

Utile pour détecter un surapprentissage éventuel et comprendre la taille de la structure de l'arbre.

### get_tree_rules(model, feature_names, ...)

Extrait et retourne les règles de décision de l'arbre sous forme de texte brut.
Aide à interpréter comment le modèle prend ses décisions.

### tune_dt_hyperparameters(...)

Effectue une recherche par grille sur un ensemble prédéfini d'hyperparamètres de l'arbre de décision.
Retourne :

* meilleurs paramètres
* meilleur score de validation croisée
* l'objet GridSearchCV ajusté

Utilisé pour trouver la profondeur optimale, les règles de séparation et les paramètres d'utilisation des variables.

### prune_tree(model, X_val, y_val, ...)

Effectue une élagage par complexité-coût à l'aide de données de validation.
Teste plusieurs valeurs de ccp_alpha, trouve la meilleure et retourne un nouveau DecisionTreeClassifier élagué.

Utile pour réduire le surapprentissage et simplifier l'arbre.

### compare_tree_depths(...)

Entraîne plusieurs arbres de décision avec différentes valeurs de max_depth et les évalue par validation croisée.
Retourne la liste des profondeurs, les scores moyens, les écarts-types et la meilleure profondeur trouvée.

Pratique pour sélectionner une profondeur maximale raisonnable avant l'entraînement final.




