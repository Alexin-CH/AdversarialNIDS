import os
import sys
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

current_dir = os.getcwd()
sys.path.append(current_dir)

from scripts.logger import SimpleLogger

def data_encoding(data, attack_encoder="label", logger=SimpleLogger()):
    """
    Encode the 'Label' column of the dataset into attack types and numerical labels.
    Args:
        data (pd.DataFrame): The DataFrame containing the 'Label' column.
        logger: Optional logger instance for tracking encoding steps.
    Returns:
        pd.DataFrame: The DataFrame with encoded attack types and numerical labels.
    """
    available_encoders = {
        "label": LabelEncoder(),
        "onehot": OneHotEncoder(sparse_output=False)
    }

    if attack_encoder not in available_encoders:
        logger.error(f"Encoder '{attack_encoder}' is not recognized.")
        logger.error(f"Available encoders: {list(available_encoders.keys())}")
        raise ValueError(f"Encoder '{attack_encoder}' is not recognized.")

    try:
        # Creating a dictionary that maps each label to its attack type
        attack_map = {
            'BENIGN': 'BENIGN',
            'DDoS': 'DDoS',
            'DoS Hulk': 'DoS',
            'DoS GoldenEye': 'DoS',
            'DoS slowloris': 'DoS',
            'DoS Slowhttptest': 'DoS',
            'PortScan': 'Port Scan',
            'FTP-Patator': 'Brute Force',
            'SSH-Patator': 'Brute Force',
            'Bot': 'Bot',
            'Web Attack Brute Force': 'Web Attack',
            'Web Attack XSS': 'Web Attack',
            'Web Attack Sql Injection': 'Web Attack',
            'Infiltration': 'Infiltration',
            'Heartbleed': 'Heartbleed'
        }

        # Creating a new column 'Attack Type' in the DataFrame based on the attack_map dictionary
        data['Attack Type'] = data['Label'].map(attack_map)
        data.drop('Label', axis = 1, inplace = True)

        encoder = available_encoders[attack_encoder]

        if attack_encoder == "onehot":
            is_attack = encoder.fit_transform((data['Attack Type'] != 'BENIGN').values.ravel().reshape(-1, 1))
            attack_classes = encoder.fit_transform(data['Attack Type'].values.ravel().reshape(-1, 1))
        else: # label encoding
            is_attack = encoder.fit_transform((data['Attack Type'] != 'BENIGN').values.ravel())
            attack_classes = encoder.fit_transform(data['Attack Type'].values.ravel())

        if logger:
            logger.debug("Data Labels after encoding:")
            for attack_type, count in data['Attack Type'].value_counts().items():
                logger.debug(f"  {attack_type}: {count}")

        logger.info(f"Attack labels encoded using {encoder} encoder.")        
        return data, is_attack, attack_classes

    except KeyError as e:
        if logger:
            logger.error(f"KeyError during data encoding: {e}")
        raise