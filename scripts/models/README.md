# Module Utilitaires de Modélisation

Utilitaires partagés pour le prétraitement, l'évaluation, le diagnostic et la gestion des variables pour tous les modèles de machine learning.

## Aperçu des Fonctions
### evaluate_model(...)
Calcule l'exactitude, les prédictions, le rapport de classification et la matrice de confusion sur les données de test.
Retourne un dictionnaire de métriques.

### check_data_leakage(...)
Effectue des diagnostics pour détecter les doublons, les variables corrélées à la cible, le déséquilibre de classes et les variables constantes.
Retourne un dictionnaire de diagnostics.

### remove_low_variance_features(...)
Supprime les variables dont la variance est inférieure à un seuil.
Retourne le DataFrame filtré et la liste des variables supprimées.

### prepare_data(...)
Pipeline complet de prétraitement incluant la suppression des variables de fuite, la conversion numérique, la gestion des valeurs manquantes et le filtrage des faibles variances.
Retourne X, y et les listes de variables supprimées.

### get_feature_importance(...)
Extrait feature_importances_ des modèles à base d'arbres ou des pipelines qui en contiennent.
Retourne la liste triée des variables les plus importantes.

### standardize_features(...)
Ajuste un StandardScaler et transforme les ensembles d'entraînement (et éventuellement de test).
Retourne les ensembles mis à l'échelle et le scaler.

### select_k_best_features(...)
Sélectionne les k meilleures variables selon le score ANOVA F.
Retourne les ensembles transformés, le sélecteur et les indices des variables sélectionnées.

### balance_classes_info(...)
Calcule les effectifs, pourcentages, ratio de déséquilibre et classes majoritaire/minoritaire.
Retourne un dictionnaire d'informations sur l'équilibre des classes.

### remove_rare_classes(...)
Filtre les classes ayant moins qu'un nombre minimal d'échantillons.
Retourne les données filtrées et la liste des classes supprimées.

### print_performance_summary(...)
Formate les métriques de validation croisée et de test dans un résumé texte avec analyse d'écart.
Affiche ou journalise le résumé.
