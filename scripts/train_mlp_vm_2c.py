import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

import matplotlib.pyplot as plt
from tqdm import tqdm

root_dir = os.getcwd().split("AdversarialNIDS")[0] + "AdversarialNIDS"
sys.path.append(root_dir)

from scripts.logger import LoggerManager
from scripts.analysis.model_analysis import perform_model_analysis

from CICIDS2017.preprocessing.dataset import CICIDS2017
from UNSWNB15.preprocessing.dataset import UNSWNB15

from scripts.models.pytorch.MLP import NetworkIntrusionMLP
from scripts.models.pytorch.CNN import NetworkIntrustionCNN
from scripts.models.pytorch.LSTM import NetworkIntrusionLSTM

from scripts.models.pytorch.train import train
from scripts.models.pytorch.visualization import display_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

lm = LoggerManager(log_dir=f"{root_dir}/logs", log_name="TDM")
logger = lm.get_logger()
title = lm.get_title()
logger.info(f"Logger initialized for '{title}'")
logger.info(f"Using device: {device}")

full_dataset = CICIDS2017( # [UNSWNB15() or CICIDS2017()]
    dataset_size="small",
    logger=logger
).optimize_memory().encode(attack_encoder="label").scale(scaler="minmax")

dataset, multiclass = full_dataset.subset(size=400*1000, multi_class=False)

X_train, X_val, y_train, y_val = dataset.split(
    one_hot=True,
    apply_smote=True,
    to_tensor=True
)

del full_dataset, dataset  # Free up memory

# Create DataLoaders
train_dataset = TensorDataset(X_train.to(device), y_train.to(device))
val_dataset = TensorDataset(X_val.to(device), y_val.to(device))

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

input_size = train_loader.dataset.tensors[0].shape[1]
num_classes = train_loader.dataset.tensors[1].shape[1]
print(f"Input size: {input_size}, Num classes: {num_classes}")

model_type = f"{input_size}x{num_classes}"

criterion = nn.CrossEntropyLoss()

model_mlp = NetworkIntrusionMLP(input_size=input_size, num_classes=num_classes).to(device)
logger.info(f"MLP Model initialized with {model_mlp.num_params()} parameters")

learning_rate_mlp = 1e-2
num_epochs_mlp = 1000

mlp_title = f"MLP_{model_type}_{num_epochs_mlp}"

optimizer_mlp = optim.AdamW(model_mlp.parameters(), lr=learning_rate_mlp)
scheduler_mlp = optim.lr_scheduler.ReduceLROnPlateau(optimizer_mlp, mode='min', factor=0.9, patience=50, min_lr=1e-8)

model_mlp, train_losses_mlp, val_losses_mlp = train(
    model=model_mlp,
    optimizer=optimizer_mlp,
    scheduler=scheduler_mlp,
    criterion=criterion,
    num_epochs=num_epochs_mlp,
    train_loader=train_loader,
    val_loader=val_loader,
    title=f"{title}_{mlp_title}",
    dir=root_dir,
    device=device,
    logger=logger
)

display_loss(
    list_epoch_loss=train_losses_mlp,
    list_val_loss=val_losses_mlp,
    title=f"{title}_{mlp_title}",
    dir=root_dir,
    plot=False,
    logger=logger,
    epoch_min=2
)

cm, cr = perform_model_analysis(
    model=model_mlp,
    X_test=X_val,
    y_test=y_val,
    logger=logger,
    model_name=f"{title}_{mlp_title}",
    dir=root_dir,
    plot=False,
    device=device
)
