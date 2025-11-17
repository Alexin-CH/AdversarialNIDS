import os
import sys
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

current_dir = os.getcwd()
sys.path.append(current_dir)

from scripts.logger import SimpleLogger

def data_encoding_UN(data, attack_encoder="label", logger=SimpleLogger()):
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
        columns = [
        "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
        "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
        "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb",
        "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit",
        "Stime", "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat",
        "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
        "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
        "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label"]
        data.columns = columns
        categorical_cols = ["proto", "state", "service","attack_cat"]
        encoders = {col: LabelEncoder().fit(data[col]) for col in categorical_cols}
        classes = encoder.classes_
        for col, encoder in encoders.items():
            data[col] = encoder.transform(data[col])
            label_to_int = {label: idx for idx, label in enumerate(classes)}
            int_to_label = {idx: label for idx, label in enumerate(classes)}
            print(f"--- Colonne : {col} ---")
            print("label_to_int =", label_to_int)
            print("int_to_label =", int_to_label)
            print()            
        # Creating a new column 'Attack Type' in the DataFrame based on the attack_map dictionary
        data.drop('label', axis = 1, inplace = True)

        encoder = available_encoders[attack_encoder]

        if attack_encoder == "onehot":
            is_attack = encoder.fit_transform((data["attack_cat"] != "").values.ravel().reshape(-1, 1))
            attack_classes = encoder.fit_transform(data["attack_cat"].values.ravel().reshape(-1, 1))
        else: # label encoding
            is_attack = encoder.fit_transform((data["attack_cat"] != "").values.ravel())
            attack_classes = encoder.fit_transform(data["attack_cat"].values.ravel())

        if logger:
            logger.debug("Data Labels after encoding:")
            for attack_type, count in data["attack_cat"].value_counts().items():
                logger.debug(f"  {attack_type}: {count}")

        logger.info(f"Attack labels encoded using {encoder} encoder.")        
        return data, is_attack, attack_classes

    except KeyError as e:
        if logger:
            logger.error(f"KeyError during data encoding: {e}")
        raise