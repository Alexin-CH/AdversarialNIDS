# AdversarialNIDS

## Description
AdversarialNIDS est un projet visant à étudier les vulnérabilités des systèmes de détection d'intrusions en réseau (NIDS) face aux techniques d'attaque adversariales. En générant des perturbations mineures mais ciblées dans les caractéristiques du trafic réseau, nous cherchons à comprendre comment tromper ces modèles d'apprentissage automatique et à évaluer leur robustesse.

## Objectif
L'objectif principal de ce projet est de :
- Évaluer la performance d'un modèle NIDS avant et après des attaques adversariales.
- Visualiser les perturbations et analyser leur impact sur les taux de détection.

## Public Ciblé
- **Alice** : Analyste en cybersécurité souhaitant évaluer la robustesse de son IDS basé sur le machine learning.
- **Bob** : Pentesteur cherchant à comprendre et tester les techniques adversariales dans un environnement contrôlé.

## Fonctionnalités
- Entraînement d'un modèle simple de détection d'intrusions (Random Forest ou réseau neuronal léger) sur un jeu de données public (CICIDS2017, UNSW-NB15).
- Génération d'exemples adversariaux en modifiant les caractéristiques du trafic.
- Évaluation de la performance du modèle (taux de détection, faux négatifs) avant et après l'attaque.
- Visualisation des perturbations et des différences de comportement du modèle.

## Contraintes Techniques
- Utilisation exclusivement de jeux de données publics et simulés.
- Les perturbations doivent rester réalistes pour conserver des traces plausibles de trafic.
- Utilisation de bibliothèques Python pour le machine learning (scikit-learn, PyTorch).

## Installation

Clonez le dépôt :
```bash
git clone https://github.com/Alexin-CH/AdversarialNIDS.git
cd AdversarialNIDS
```

Installez les dépendances :
```bash
pip install -r requirements.txt
```

