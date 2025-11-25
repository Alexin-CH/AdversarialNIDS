import torch

def recompute_features(x_adv):
    dur = x_adv[:, CICIDS2017_DICT['Flow Duration']]
    fwd_pkts = x_adv[:, CICIDS2017_DICT['Total Fwd Packets']]
    bwd_pkts = x_adv[:, CICIDS2017_DICT['Total Backward Packets']]
    fwd_bytes = x_adv[:, CICIDS2017_DICT['Total Length of Fwd Packets']]
    bwd_bytes = x_adv[:, CICIDS2017_DICT['Total Length of Bwd Packets']]
    
    # safe versions
    # dur_s = torch.clamp(dur, min=1e-6)
    if torch.any(dur <= 0):
        print("Warning: Flow Duration has non-positive values, clamping to avoid division by zero.")
        dur_s = torch.clamp(dur, min=1e-6)
    else:
        dur_s = dur
    fwd_pkts_s = torch.clamp(fwd_pkts, min=1)
    bwd_pkts_s = torch.clamp(bwd_pkts, min=1)

    # Flow-level
    x_adv[:, CICIDS2017_DICT['Flow Bytes/s']]    = (fwd_bytes + bwd_bytes) / dur_s
    x_adv[:, CICIDS2017_DICT['Flow Packets/s']]  = (fwd_pkts + bwd_pkts) / dur_s

    # Average segment size
    x_adv[:, CICIDS2017_DICT['Avg Fwd Segment Size']] = fwd_bytes / fwd_pkts_s
    x_adv[:, CICIDS2017_DICT['Avg Bwd Segment Size']] = bwd_bytes / bwd_pkts_s

    # Fwd/Bwd IAT Mean
    x_adv[:, CICIDS2017_DICT['Fwd IAT Mean']] = x_adv[:, CICIDS2017_DICT['Fwd IAT Total']] / fwd_pkts_s
    x_adv[:, CICIDS2017_DICT['Bwd IAT Mean']] = x_adv[:, CICIDS2017_DICT['Bwd IAT Total']] / bwd_pkts_s

    # Packet length global metrics
    x_adv[:, CICIDS2017_DICT['Min Packet Length']] = torch.min(
        x_adv[:, CICIDS2017_DICT['Fwd Packet Length Min']],
        x_adv[:, CICIDS2017_DICT['Bwd Packet Length Min']]
    )

    x_adv[:, CICIDS2017_DICT['Max Packet Length']] = torch.max(
        x_adv[:, CICIDS2017_DICT['Fwd Packet Length Max']],
        x_adv[:, CICIDS2017_DICT['Bwd Packet Length Max']]
    )

    return x_adv


FIRST_LEVEL_FEATURES = [
    'Destination Port',
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    'Fwd Packet Length Max',
    'Fwd Packet Length Min',
    'Fwd Packet Length Mean',
    'Fwd Packet Length Std',
    'Bwd Packet Length Max',
    'Bwd Packet Length Min',
    'Bwd Packet Length Mean',
    'Bwd Packet Length Std',
    'Fwd IAT Total',
    'Fwd IAT Max',
    'Fwd IAT Min',
    'Bwd IAT Total',
    'Bwd IAT Max',
    'Bwd IAT Min',
    'Fwd PSH Flags',
    'Fwd URG Flags',
    'Fwd Header Length',
    'Bwd Header Length',
    'FIN Flag Count',
    'SYN Flag Count',
    'RST Flag Count',
    'PSH Flag Count',
    'ACK Flag Count',
    'URG Flag Count',
    'CWE Flag Count',
    'ECE Flag Count',
    'Fwd Header Length.1',
    'Subflow Fwd Packets',
    'Subflow Fwd Bytes',
    'Subflow Bwd Packets',
    'Subflow Bwd Bytes',
    'Init_Win_bytes_forward',
    'Init_Win_bytes_backward',
    'act_data_pkt_fwd',
    'min_seg_size_forward'
]

CICIDS2017_DICT = {
    'Destination Port': 0,
    'Flow Duration': 1,
    'Total Fwd Packets': 2,
    'Total Backward Packets': 3,
    'Total Length of Fwd Packets': 4,
    'Total Length of Bwd Packets': 5,
    'Fwd Packet Length Max': 6,
    'Fwd Packet Length Min': 7,
    'Fwd Packet Length Mean': 8,
    'Fwd Packet Length Std': 9,
    'Bwd Packet Length Max': 10,
    'Bwd Packet Length Min': 11,
    'Bwd Packet Length Mean': 12,
    'Bwd Packet Length Std': 13,
    'Flow Bytes/s': 14,
    'Flow Packets/s': 15,
    'Flow IAT Mean': 16,
    'Flow IAT Std': 17,
    'Flow IAT Max': 18,
    'Flow IAT Min': 19,
    'Fwd IAT Total': 20,
    'Fwd IAT Mean': 21,
    'Fwd IAT Std': 22,
    'Fwd IAT Max': 23,
    'Fwd IAT Min': 24,
    'Bwd IAT Total': 25,
    'Bwd IAT Mean': 26,
    'Bwd IAT Std': 27,
    'Bwd IAT Max': 28,
    'Bwd IAT Min': 29,
    'Fwd PSH Flags': 30,
    'Fwd URG Flags': 31,
    'Fwd Header Length': 32,
    'Bwd Header Length': 33,
    'Fwd Packets/s': 34,
    'Bwd Packets/s': 35,
    'Min Packet Length': 36,
    'Max Packet Length': 37,
    'Packet Length Mean': 38,
    'Packet Length Std': 39,
    'Packet Length Variance': 40,
    'FIN Flag Count': 41,
    'SYN Flag Count': 42,
    'RST Flag Count': 43,
    'PSH Flag Count': 44,
    'ACK Flag Count': 45,
    'URG Flag Count': 46,
    'CWE Flag Count': 47,
    'ECE Flag Count': 48,
    'Down/Up Ratio': 49,
    'Average Packet Size': 50,
    'Avg Fwd Segment Size': 51,
    'Avg Bwd Segment Size': 52,
    'Fwd Header Length.1': 53,
    'Subflow Fwd Packets': 54,
    'Subflow Fwd Bytes': 55,
    'Subflow Bwd Packets': 56,
    'Subflow Bwd Bytes': 57,
    'Init_Win_bytes_forward': 58,
    'Init_Win_bytes_backward': 59,
    'act_data_pkt_fwd': 60,
    'min_seg_size_forward': 61,
    'Active Mean': 62,
    'Active Std': 63,
    'Active Max': 64,
    'Active Min': 65,
    'Idle Mean': 66,
    'Idle Std': 67,
    'Idle Max': 68,
    'Idle Min': 69
}

MODIFIABLE_FEATURES = [CICIDS2017_DICT[feat] for feat in FIRST_LEVEL_FEATURES]

