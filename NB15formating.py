import pandas as pd
import torch as pt 
from sklearn.preprocessing import LabelEncoder

#Import of the dataset
df = pd.read_csv(r"C:\msys64\home\valen\TDpython\UNSW-NB15_1(1).csv" ,sep=",",header=None)
#Label of the colums for thi dataset
columns = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
    "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
    "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit",
    "Stime", "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
    "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label"
]
df.columns = columns

df = df.fillna("unknown")

categorical_cols = ["proto", "state", "service"]
encoders = {col: LabelEncoder().fit(df[col]) for col in categorical_cols}

for col, encoder in encoders.items():
    df[col] = encoder.transform(df[col])
#Use of one hot encoding pour the categori column   
df = pd.get_dummies(df, columns=["attack_cat"])

#Save path
output_path = r"C:\msys64\home\valen\TDpython\UNSW-NB15_1_formatted.csv"

df.to_csv(output_path, index=False, header=False)

print(f"✅ Fichier formaté sauvegardé : {output_path}")
print(f"Nombre de lignes : {len(df)}, nombre de colonnes : {len(df.columns)}")