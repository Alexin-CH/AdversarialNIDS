import torch

def compute_features(x_adv):
    dur    = x_adv[:, UNSWNB15_DICT['Flow Duration']]
    sbytes = x_adv[:, UNSWNB15_DICT['Total Length of Fwd Packets']]
    dbytes = x_adv[:, UNSWNB15_DICT['Total Length of Bwd Packets']]
    spkts  = x_adv[:, UNSWNB15_DICT['Total Fwd Packets']]
    dpkts  = x_adv[:, UNSWNB15_DICT['Total Backward Packets']]

    dur_safe   = torch.clamp(dur, min=1e-6)
    spkts_safe = torch.clamp(spkts, min=1)
    dpkts_safe = torch.clamp(dpkts, min=1)

    x_adv[:, UNSWNB15_DICT['Fwd Packets/s']]       = spkts / dur_safe
    x_adv[:, UNSWNB15_DICT['Bwd Packets/s']]       = dpkts / dur_safe
    x_adv[:, UNSWNB15_DICT['Avg Fwd Segment Size']] = sbytes / spkts_safe
    x_adv[:, UNSWNB15_DICT['Avg Bwd Segment Size']] = dbytes / dpkts_safe
    x_adv[:, UNSWNB15_DICT['Sintpkt']]             = dur_safe / (spkts_safe - 1 + eps)
    x_adv[:, UNSWNB15_DICT['Dintpkt']]             = dur_safe / (dpkts_safe - 1 + eps)

    return x_adv

FIRST_LEVEL_FEATURES = [
    "srcip",
    "sport",
    "dstip",
    "dsport",
    "proto",
    "service",

    "dur",

    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",

    "sttl",
    "dttl",

    "swin",
    "dwin",

    "state",

    "synack",
    "ackdat",
    "tcprtt",

    "sloss",
    "dloss",

    "trans_depth",
    "res_bdy_len",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",

    "ct_srv_src",
    "ct_srv_dst",
    "ct_dst_ltm",
    "ct_src_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
]

UNSWNB15_DICT = {
    'sport': 0,
    'Destination Port': 1,
    'proto': 2,
    'state': 3,
    'Flow Duration': 4,
    'Total Length of Fwd Packets': 5,
    'Total Length of Bwd Packets': 6,
    'sttl': 7,
    'dttl': 8,
    'sloss': 9,
    'dloss': 10,
    'service': 11,
    'Fwd Packets/s': 12,
    'Bwd Packets/s': 13,
    'Total Fwd Packets': 14,
    'Total Backward Packets': 15,
    'Init_Win_bytes_forward': 16,
    'Init_Win_bytes_backward': 17,
    'stcpb': 18,
    'dtcpb': 19,
    'Avg Fwd Segment Size': 20,
    'Avg Bwd Segment Size': 21,
    'trans_depth': 22,
    'res_bdy_len': 23,
    'Sjit': 24,
    'Djit': 25,
    'Stime': 26,
    'Ltime': 27,
    'Sintpkt': 28,
    'Dintpkt': 29,
    'tcprtt': 30,
    'synack': 31,
    'ackdat': 32,
    'is_sm_ips_ports': 33,
    'ct_state_ttl': 34,
    'ct_flw_http_mthd': 35,
    'is_ftp_login': 36,
    'ct_ftp_cmd': 37,
    'ct_srv_src': 38,
    'ct_srv_dst': 39,
    'ct_dst_ltm': 40,
    'ct_src_ltm': 41,
    'ct_src_dport_ltm': 42,
    'ct_dst_sport_ltm': 43,
    'ct_dst_src_ltm': 44
}

MODIFIABLE_FEATURES = [UNSWNB15_DICT[feat] for feat in FIRST_LEVEL_FEATURES if feat in UNSWNB15_DICT]

INTEGER_FEATURES = [
    'sport',
    'Destination Port',
    'proto',
    'state',
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    'sttl',
    'dttl',
    'sloss',
    'dloss',
    'service',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Init_Win_bytes_forward',
    'Init_Win_bytes_backward',
    'stcpb',
    'dtcpb',
    'trans_depth',
    'res_bdy_len',
    'is_sm_ips_ports',
    'ct_state_ttl',
    'ct_flw_http_mthd',
    'is_ftp_login',
    'ct_ftp_cmd',
    'ct_srv_src',
    'ct_srv_dst',
    'ct_dst_ltm',
    'ct_src_ltm',
    'ct_src_dport_ltm',
    'ct_dst_sport_ltm',
    'ct_dst_src_ltm'
]

INTEGER_INDICES = [UNSWNB15_DICT[feat] for feat in INTEGER_FEATURES if feat in UNSWNB15_DICT]