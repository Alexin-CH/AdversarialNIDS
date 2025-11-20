# Module Utilitaires de Modélisation

Utilitaires partagés pour le l'évaluation, le diagnostic et la gestion des variables pour tous les modèles de machine learning.

## Aperçu des Fonctions
### check_data_leakage(...)
Effectue des diagnostics pour détecter les doublons, les variables corrélées à la cible, le déséquilibre de classes et les variables constantes.
Retourne un dictionnaire de diagnostics.

### remove_low_variance_features(...)
Supprime les variables dont la variance est inférieure à un seuil.
Retourne le DataFrame filtré et la liste des variables supprimées.

### get_feature_importance(...)
Extrait feature_importances_ des modèles à base d'arbres ou des pipelines qui en contiennent.
Retourne la liste triée des variables les plus importantes.

### select_k_best_features(...)
Sélectionne les k meilleures variables selon le score ANOVA F.
Retourne les ensembles transformés, le sélecteur et les indices des variables sélectionnées.

### balance_classes_info(...)
Calcule les effectifs, pourcentages, ratio de déséquilibre et classes majoritaire/minoritaire.
Retourne un dictionnaire d'informations sur l'équilibre des classes.

