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

### 2. Surrogate Attack
#### Principe de Fonctionnement

L'attaque Surrogate est une méthode qui permet de passer outre l'absence de gradient d'un modèle de détection. Un MLP peu profond est utilisé pour simuler la prédiction du random forest, l'attaque adversaire se fait sur ce réseau de substitut en utilisant les prédictions de ce réseau. Les données perturbée sont ensuite fournit au random forest.

#### Caractéristiques Principales

- **Adapté à tout type de modèle** : Fonctionne sans accès aux gradients du modèle cible

#### Description Étape par Étape

1. **Initialisation** : Un MLP peu profond est entrainé sur le même type de données que le randdom forest.

2. **Perturbation itérative** : L'attaque commence avec un échantillon d'entrée initial `x`.Le processus perturbe itérativement `x` dans une direction estimée vers la frontière de décision, visant la malclassification

3. **Estimation de direction** : La direction de perturbation est déterminée en utilisant une estimation non-biaisée de la direction du gradient à la frontière de décision, basée sur les prédictions du modèle

4. **Contrôle de magnitude** : La magnitude de la perturbation est soigneusement contrôlée pour que l'exemple adversarial reste proche de l'entrée originale

5. **Convergence** : Le processus continue jusqu'à une étape fixée.

### 3. Constrains
Pour avoir des attaques adverses cohérentes, on veut avoir les bornes min/max uniquement à partir des échantillons 
d’attaque afin de garantir que les exemples adverses restent réalistes et dans l’espace de distribution des attaques, 
en évitant qu’ils ne dérivent vers des plages de caractéristiques propres aux données BENIGN

```python
attack_mask = y_train != 0 #or 3 for BENIGN
X_attacks = X_train[attack_mask]
mins = torch.tensor(np.percentile(X_attacks, 1, axis=0), dtype=torch.float32).to(device)
maxs = torch.tensor(np.percentile(X_attacks, 99, axis=0), dtype=torch.float32).to(device)
```

### 3. Méthode par Substitut
#### Principe de Fonctionnement

La méthode de substitut se rapproche de l'attaque par surrogate dans son utilisation d'un MLP peu profond pour simuler le modèle attaqué. Elle permet donc également de passer outre l'absence de gradient de modèles type random forest. Elle se différencie du surrogate par utilisation des données prédites du modèle attaqué (la mise en place est donc un peu plus dur).

#### Caractéristiques Principales

- **Adapté à tout type de modèle** : Fonctionne sans accès aux gradients du modèle cible

#### Description Étape par Étape

1. **Initialisation** : Un MLP peu profond est entrainé sur des données étiquettés par le modèle attaqué.

2. **Perturbation itérative** : L'attaque commence avec un échantillon d'entrée initial `x`.Le processus perturbe itérativement `x` dans une direction estimée vers la frontière de décision, visant la malclassification

3. **Estimation de direction** : La direction de perturbation est déterminée en utilisant une estimation non-biaisée de la direction du gradient à la frontière de décision, basée sur les prédictions du modèle

4. **Contrôle de magnitude** : La magnitude de la perturbation est soigneusement contrôlée pour que l'exemple adversarial reste proche de l'entrée originale

5. **Convergence** : Le processus continue jusqu'à une étape fixée ou mauvaise classification par le modèlé substitut de tout les échantillons.
