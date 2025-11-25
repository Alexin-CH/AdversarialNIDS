# Dataset UNSWNB15 - Documentation

## Modules

Les fonctions relatives au prépraitement du dataset UNSWNB15 sont organisées dans le sous-répertoire `preprocessing/`. 

Les fonctions relatives aux analyses exploratoires et visualisations sont organisées dans le sous-répertoire `analysis/`.

### 1. `dataset.py` - Classe principale

#### Classe `UNSWNB15`

Classe principale qui encapsule l'ensemble du workflow de prétraitement.

**Constructeur :**
```python
UNSWNB15(dataset_size=None, logger=SimpleLogger())
```
- Initialise le dataset en téléchargeant et préparant les données
- `dataset_size` : Taille du dataset (`full` pour concatener les 4 CSV ou `small` pour n'utiliser que le premier)
- `logger` : Instance de logger pour le suivi des opérations

**Méthodes :**

- **`optimize_memory()`** : Optimise l'utilisation de la mémoire du dataset
  - Retourne : `self` (chaînage de méthodes)

- **`encode(attack_encoder="label")`** : Encode les labels d'attaque
  - `attack_encoder` : Type d'encodeur ("label" ou "onehot")
  - Retourne : `self` (chaînage de méthodes)
  - Crée les attributs : `self.is_attack`, `self.attack_classes`

- **`scale(scaler="standard")`** : Normalise les features du dataset
  - `scaler` : Type de normalisation ("standard" ou "minmax")
  - Retourne : `self` (chaînage de méthodes)
  - Crée l'attribut : `self.scaled_features`

- **`subset(size=None, multi_class=False)`** : Sous-échantillonne le dataset
  - `size` : Taille cible du dataset
  - `multi_class` : Si True, utilise les classes d'attaque ; sinon binaire (attaque/bénin)
  - Retourne : `self` (chaînage de méthodes)

- **`split(test_size=0.2, to_tensor=False, one_hot=False, apply_smote=False)`** : Divise en ensembles train/test
  - `test_size` : Proportion des données de test (défaut: 0.2)
  - `to_tensor` : Convertit en tenseurs PyTorch si True
  - `one_hot` : Applique l'encodage one-hot aux labels si True
  - `apply_smote` : Applique SMOTE pour équilibrer les classes si True
  - Retourne : `(X_train, X_test, y_train, y_test)`

**Exemple d'utilisation :**
```python
dataset = UNSWNB15(dataset_size="small").optimize_memory().encode().scale().subset(size=100000, multi_class=False)
X_train, X_test, y_train, y_test = dataset.split(test_size=0.2, apply_smote=True)
```

**Colonnes du dataset :**
```
Index(['sport', 'Destination Port', 'proto', 'state', 'Flow Duration',
       'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'sttl',
       'dttl', 'sloss', 'dloss', 'service', 'Fwd Packets/s', 'Bwd Packets/s',
       'Total Fwd Packets', 'Total Backward Packets', 'Init_Win_bytes_forward',
       'Init_Win_bytes_backward', 'stcpb', 'dtcpb', 'Avg Fwd Segment Size',
       'Avg Bwd Segment Size', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit',
       'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat',
       'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login',
       'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
       'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm'],
      dtype='object')
```

#### 1\. Identification & Informations de Base

| Feature | Description & Signification | Min / Max (Type) | Corrélations & Usage |
| :--- | :--- | :--- | :--- |
| **sport** | **Source Port**. Port utilisé par l'émetteur. | 0 - 65535 (Int) | Aléatoire (Ephemeral) pour les clients normaux. Fixe pour les attaquants mal configurés. |
| **Destination Port** | Port ciblé (ex: 80 HTTP, 443 HTTPS). | 0 - 65535 (Int) | Définit le **Service**. |
| **proto** | Protocole de transport (6=TCP, 17=UDP). | 0 - 255 (Int) | Détermine la structure du paquet. |
| **service** | Service applicatif détecté (ex: HTTP, SSH, FTP). | Catégorique | **Très forte** avec `Dst Port`. |
| **state** | État de la connexion (ex: FIN, CON, REQ). | Catégorique | Indique si l'attaque a abouti. |
| **Flow Duration** | Durée totale du flux en microsecondes. | 0 - Variable (Float) | **Forte** avec `Ltime - Stime`. |

#### 2\. Volume et Débit (Traffic Dynamics)

| Feature | Description & Signification | Min / Max | Corrélations Principales |
| :--- | :--- | :--- | :--- |
| **Total Fwd/Bwd Packets** | Nombre total de paquets envoyés/reçus. | 1 - Variable | **100% corrélé** avec les `Total Length` correspondants. |
| **Total Length Fwd/Bwd** | Volume total des données (payload) en Bytes. | 0 - Variable | Indicateur d'exfiltration de données (si Bwd élevé). |
| **Fwd/Bwd Packets/s** | Taux de transfert (Vitesse). | 0.0 - Variable | **Inverse** aux temps inter-arrivée (`Sintpkt`). Élevé pour DoS. |
| **Avg Fwd/Bwd Segment Size** | Taille moyenne des segments TCP. | 0 - 65535 | Même information que `Packet Mean`. |

#### 3\. Caractéristiques Techniques TCP/IP (Fingerprinting)

| Feature | Description | Min / Max | Corrélations |
| :--- | :--- | :--- | :--- |
| **sttl / dttl** | **Source/Dest Time To Live**. Durée de vie du paquet. | 0 - 255 | Indique l'OS (ex: Windows=128, Linux=64). |
| **sloss / dloss** | **Source/Dest Loss**. Paquets perdus/retransmis. | 0 - Variable | Élevé lors de congestion ou attaque DoS massive. |
| **stcpb / dtcpb** | **Source/Dest TCP Base Sequence**. Numéro de séquence de départ. | 0 - $2^{32}$ | Aléatoire. Si répétitif = Outil d'attaque mal codé. |
| **Init\_Win\_bytes\_fwd/bwd** | Taille de la fenêtre TCP initiale. | 0 - 65535 | Signature forte du client/navigateur. |
| **tcprtt** | **TCP Round Trip Time**. Temps aller-retour ($RTT = SynAck + AckDat$). | 0.0 - Variable | Somme parfaite de `synack` et `ackdat`. |
| **synack** | Temps entre le paquet SYN et le SYN-ACK. | 0.0 - Variable | Mesure la latence serveur. |
| **ackdat** | Temps entre le SYN-ACK et le ACK final (Data). | 0.0 - Variable | Mesure la latence client. |

#### 4\. Dynamique Temporelle (Jitter & Temps)

| Feature | Description | Min / Max | Corrélations |
| :--- | :--- | :--- | :--- |
| **Sjit / Djit** | **Source/Dest Jitter**. Variation du délai entre paquets. | 0.0 - Variable | Élevé pour streaming/VoIP, faible pour transfert de fichier. |
| **Sintpkt / Dintpkt** | **Inter-Packet Time**. Temps moyen entre deux paquets. | 0.0 - Variable | **Inverse** du débit (`Packets/s`). |
| **Stime / Ltime** | **Start/Last Time**. Timestamp début et fin. | Unix Timestamp | `Ltime - Stime` $\approx$ `Flow Duration`. |

#### 5\. Comportement Applicatif & Anomalies (Deep Packet Inspection)

| Feature | Description | Valeurs | Usage Sécurité |
| :--- | :--- | :--- | :--- |
| **trans\_depth** | Profondeur de transaction (Pipelining HTTP). | 0 - N | Compte les requêtes imbriquées (HTTP Keep-Alive). |
| **res\_bdy\_len** | Taille du corps de la réponse (Response Body). | 0 - Variable | Taille du fichier téléchargé. |
| **is\_sm\_ips\_ports** | **Land Attack**. (1 si IP Src=Dst ET Port Src=Dst). | 0 ou 1 | **Alerte Rouge** si 1 (Attaque DoS ancienne). |
| **is\_ftp\_login** | Indique si le flux contient un login FTP. | 0 ou 1 | Pour détecter brute-force FTP. |
| **ct\_ftp\_cmd** | Compteur de commandes FTP (ex: USER, PASS). | 0 - N | Si \> 0 et `is_ftp_login`=0 : Commandes sans auth. |
| **ct\_flw\_http\_mthd** | Compteur de méthodes HTTP (GET, POST...). | 0 - N | Flux HTTP valides. |

#### 6\. Compteurs Statistiques (Les "Count Features")

| Feature | Description (Sur les 100 dernières connexions) | Usage / Corrélation |
| :--- | :--- | :--- |
| **ct\_srv\_src** | Nb de connexions vers le même Service (port) depuis cette Source IP. | Détecte scan de port ou brute-force spécifique. |
| **ct\_srv\_dst** | Nb de connexions vers le même Service (port) vers cette Dest IP. | Détecte attaque DoS sur un service (ex: HTTP flood). |
| **ct\_dst\_ltm** | Nb de connexions vers cette Dest IP (Last Time Mode). | Détecte DDoS (IP inondée). |
| **ct\_src\_ltm** | Nb de connexions depuis cette Source IP. | Détecte machine infectée (Botnet). |
| **ct\_src\_dport\_ltm** | Nb de connexions IP Source $\to$ Port Dest. | Scan de port ciblé. |
| **ct\_dst\_sport\_ltm** | Nb de connexions IP Dest $\to$ Port Source. | Retour de scan ou réponse massive. |
| **ct\_dst\_src\_ltm** | Nb de connexions entre cette paire IP Src/Dst exactes. | Acharnement sur une cible précise. |
| **ct\_state\_ttl** | Corrélation étrange propre à CICFlowMeter (State vs TTL). | Souvent bruitée, moins interprétable. |
