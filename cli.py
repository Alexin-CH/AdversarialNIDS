from datetime import datetime
import os
import pandas as pd
import numpy as np

from CICIDS2017.preprocessing.dataset import CICIDS2017

from scripts.logger import LoggerManager

from scripts.models.decision_tree import train_decision_tree
from scripts.models.random_forest import train_random_forest
from scripts.models.knn import train_knn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils import resample


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def prompt_menu(options, title=None):
    if title:
        print("\n"+"=="*15 + f" {title} " + "=="*15)
    for i, opt in enumerate(options, start=1):
        print(f"{i}. {opt}")
    print("0. Retour")
    choice = input("Choix: ").strip()
    try:
        return int(choice)
    except ValueError:
        return -1

def load_previous_results(logger):
    """List CSV files in RESULTS_DIR and let the user choose"""
    files = sorted([f for f in os.listdir(RESULTS_DIR) if f.lower().endswith('.csv')])
    if not files:
        logger.info('Aucun fichier de résultats dans %s', RESULTS_DIR)
        return None
    opts = files + ['Entrer un chemin personnalisé']
    choice = prompt_menu(opts, title='Choisir un fichier de résultats')
    if choice == 0:
        return None
    if 1 <= choice <= len(files):
        path = os.path.join(RESULTS_DIR, files[choice - 1])
    else:
        path = input('Chemin vers le fichier CSV de résultats précédents: ').strip()
    if not os.path.exists(path):
        logger.error('Fichier introuvable: %s', path)
        return None
    try:
        df = pd.read_csv(path)
        logger.info('Chargé %d lignes depuis %s', len(df), path)
        print(df.head())
        return df
    except Exception as e:
        logger.error('Erreur lecture CSV: %s', e)
        return None

def select_dataset(logger):
    options = ["CICIDS2017", "UNSW-NB15"]
    choice = prompt_menu(options, title='Choisissez le dataset')
    if choice == 0:
        return None
    if choice == 1:
        logger.info('Chargement de CICIDS2017 (prétraitement en cours si nécessaire)')
        ds = CICIDS2017(logger=logger).encode(attack_encoder="label").optimize_memory().scale(scaler="minmax")
        logger.info('Dataset CICIDS2017 chargé: %d lignes x %d colonnes', ds.data.shape[0], ds.data.shape[1])
        # Return the dataset object so callers can access scaled_features / attack_classes
        return ds
    if choice == 2:
        print('#TODO')
        logger.info('UNSW-NB15 dataset selectionné mais non implémenté')
        return None
    logger.info('Choix invalide, retour')
    return None

def inspect_basic(df):
    print('\n--- Aperçu basique ---')
    print('Shape:', df.shape)
    print('\nTypes:')
    print(df.dtypes)
    print('\nDistribution des types d\'attaque:')
    print(df['Attack Type'].value_counts())

    
    num = df.select_dtypes(include=[np.number]) # Ne prendre que les colonnes numériques
    vars = num.var().sort_values(ascending=False)
    top_features = list(vars.head(10).index)
    print('\nTop 10 features par variance (fallback):')
    for f in top_features:
        print(f)

    # Afficher le describe pour ces features
    print('\nDescription (numérique) pour les 10 features sélectionnées:')
    print(num[top_features].describe().T)


def inspect_advanced(df):
    raise NotImplementedError('#TODO: Implement advanced inspection here')

def train_model_with_dataset(X, y, model_key, logger=None):
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    #if model_key == 'DecisionTree':
    #    model, cv = train_decision_tree(X_train, y_train, max_depth=6, logger=logger)
    #elif model_key == 'RandomForest':
    #    model, cv = train_random_forest(X_train, y_train, n_estimators=50, max_depth=10, logger=logger)
    #elif model_key == 'KNN':
    #    model, cv = train_knn(X_train, y_train, n_neighbors=5, logger=logger)
    raise NotImplementedError('#TODO: Implement model training here')

def run_attack_simulation(*args, **kwargs):
    # Attaques non implémentées dans cette CLI (sera fait a posteriori par l'équipe).
    raise NotImplementedError('#TODO: Implement attack simulations here')

def main():
    mgr = LoggerManager(log_dir='logs', log_name='cli')
    logger = mgr.get_logger()
    logger.info('Starting CLI')

    while True:
        main_opts = ['Charger des résultats précédents (CSV)', 'Lancer une nouvelle session/run']
        choice = prompt_menu(main_opts, title='AdversarialNIDS CLI')
        if choice == 0:
            break
        if choice == 1:
            load_previous_results(logger)
            continue
        if choice == 2:
            ds = select_dataset(logger)
            if ds is None:
                continue
            sub_opts = ['Oui', 'Non']
            sub_choice = prompt_menu(sub_opts, title='Sous-échantillonner le dataset pour accélérer ?')
            subsampled = False
            sample_size = None
            # df_work is the DataFrame view used by inspection/training when subsampled
            df_work = ds.data
            if sub_choice == 1:
                # ask for sample size
                try:
                    raw = input('Taille de l\'échantillon (par défaut 5000): ').strip()
                    sample_size = int(raw) if raw else 5000
                except ValueError:
                    sample_size = 5000
                sample_size = min(sample_size, len(ds.data))
                df_work = ds.data.sample(n=sample_size, random_state=0)
                subsampled = True
                logger.info('Dataset sous-échantillonné à %d lignes', sample_size)

            # dataset menu
            while True:
                ds_opts = ['Afficher informations basiques', 'Analyse avancée', 'Modélisation']
                ch = prompt_menu(ds_opts, title='Dataset menu')
                if ch == 0:
                    break
                if ch == 1:
                    inspect_basic(df_work)
                    continue
                if ch == 2:
                    inspect_advanced(df_work)
                    continue
                if ch == 3:
                    # modeling menu
                    model_opts = ['DecisionTree', 'RandomForest', 'KNN']
                    m_choice = prompt_menu(model_opts, title='Choisir un modèle')
                    if m_choice == 0:
                        continue
                    if 1 <= m_choice <= len(model_opts):
                        model_name = model_opts[m_choice - 1]
                    else:
                        logger.info('Choix modèle invalide')
                        continue
                    continue
        logger.info('Choix invalide')

    logger.info('CLI terminé')


if __name__ == '__main__':
    main()
