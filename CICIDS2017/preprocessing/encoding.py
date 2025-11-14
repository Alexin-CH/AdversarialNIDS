import pandas as pd
from sklearn.preprocessing import LabelEncoder

def data_encoding(data, logger=None):
    """
    Encode the 'Label' column of the dataset into attack types and numerical labels.
    Args:
        data (pd.DataFrame): The DataFrame containing the 'Label' column.
        logger: Optional logger instance for tracking encoding steps.
    Returns:
        pd.DataFrame: The DataFrame with encoded attack types and numerical labels.
    """
    if logger:
        logger.info("Starting data encoding process")
        logger.info("Data Labels before encoding:")
        for label, count in data['Label'].value_counts().items():
            logger.info(f"  {label}: {count}")

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

        le = LabelEncoder()
        data['Attack Number'] = le.fit_transform(data['Attack Type'])

        if logger:
            logger.info("Data Labels after encoding:")
            for attack_type, count in data['Attack Type'].value_counts().items():
                logger.info(f"  {attack_type}: {count}")
        
        return data

    except KeyError as e:
        if logger:
            logger.error(f"KeyError during data encoding: {e}")
        raise