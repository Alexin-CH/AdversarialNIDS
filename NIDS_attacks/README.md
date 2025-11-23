# NIDS Attacks - Adversarial Examples Implementation

Ce dossier contient l'implémentation de différentes techniques d'attaques adversariales contre les systèmes de détection d'intrusions réseau (NIDS). Ces attaques visent à tromper les modèles de machine learning en créant des exemples adversariaux qui sont malclassifiés par les systèmes de détection.

## Objectif

L'objectif principal est d'évaluer la robustesse des modèles NIDS contre différents types d'attaques adversariales et de comprendre leurs vulnérabilités pour améliorer leur sécurité.

## Types d'Attaques Implémentées

### 1. HopSkipJump Attack

#### Principe de Fonctionnement

L'attaque HopSkipJump est une méthode sophistiquée qui ne nécessite pas l'accès aux gradients du modèle cible, ce qui la rend applicable à une large gamme de modèles, y compris les modèles non-différentiables comme les arbres de décision.

#### Caractéristiques Principales

- **Sans gradient (Gradient-free)** : Fonctionne sans accès aux gradients du modèle cible
- **Efficace en requêtes** : Minimise le nombre d'interrogations du modèle

#### Description Étape par Étape

1. **Initialisation** : L'attaque commence avec un échantillon d'entrée initial `x`

2. **Perturbation itérative** : Le processus perturbe itérativement `x` dans une direction estimée vers la frontière de décision, visant la malclassification

3. **Estimation de direction** : La direction de perturbation est déterminée en utilisant une estimation non-biaisée de la direction du gradient à la frontière de décision, basée sur les prédictions du modèle

4. **Contrôle de magnitude** : La magnitude de la perturbation est soigneusement contrôlée pour que l'exemple adversarial reste proche de l'entrée originale

5. **Convergence** : Le processus continue jusqu'à ce que l'entrée perturbée réussisse à tromper le modèle cible

### Configuration des Paramètres

Les scripts permettent de configurer plusieurs paramètres :

- **Dataset** : `"CICIDS2017"` ou `"UNSWNB15"`
- **Nombre d'échantillons** : Nombre d'exemples à attaquer
- **Taille du dataset d'entrainement** : Taille du sous-dataset servant à l'entrainement du modèle
- **Visualisation** : Affichage détaillé par échantillon

```python
results = [dt/rf]_hopskipjump_attack(
    dataset="CICIDS2017",
    ds_train_size = 10000,
    nb_samples=10,
    per_sample_visualization=True
)
```