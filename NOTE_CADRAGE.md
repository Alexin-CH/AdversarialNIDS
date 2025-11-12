# Note de Cadrage : Attaques Adversariales sur les Systèmes de Détection d’Intrusion

## Informations Générales
- **Projet** : Attaques adversariales sur les systèmes de détection d’intrusion basés sur l’apprentissage automatique
- **Date** : 12/11/2025
- **Version** : 1.0

## But et Objectifs
- Étudier comment des attaquants peuvent exploiter les failles des modèles d’apprentissage automatique utilisés pour la détection d’intrusions (NIDS).
- Entraînement de modèles simples de détection d’intrusion (ex. Random Forest ou réseau neuronal léger) sur des jeux de données publics (CICIDS2017, UNSW-NB15).
- Génération d’exemples adversariaux en modifiant légèrement les caractéristiques du trafic (feature perturbation).
- Évaluation de la performance du modèle avant et après attaque (taux de détection, faux négatifs).
- Visualisation des perturbations et des différences de comportement du modèle.
- Fine tuning des modèles afin de résister aux attaques.

## Périmètre du Projet
- **Inclus** : Mise en place de détection d’attaque réseaux par machine learning et des attaques adverseriales.
- **Exclus** : Analyse de cybersécurité des réseaux.

## Livrables Attendus
- Plusieurs algorithmes de machine learning de détection.
- Algorithme d’attaque.
- Documentation.

## Planning Prévisionnel
- **Premier sprint (12/11 - 19/11)** : Mise en place des algorithmes de détections.
- **Deuxième sprint (19/11 - 26/11)** : Mise en place de l’algorithme d’attaque.
- **Troisième sprint (26/11 - 1/12)** : Fine tuning des modèles.

## Risques et Contraintes
- Utiliser exclusively des jeux de données publics et simulés, pas de trafic réel de production.
- Les perturbations doivent rester réalistes (conserver des traces plausibles de trafic).
- Utilisation obligatoire de bibliothèques Python pour le machine learning (scikit-learn, PyTorch).

## Validation
- **Client** : Oui