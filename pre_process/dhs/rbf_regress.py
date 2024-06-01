import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.optim as optim

from datetime import datetime

import optuna 
from optuna.pruners import MedianPruner
from optuna.trial import TrialState
import logging
from utils import save_checkpoint, load_checkpoint, RBFSpatialRelationPositionEncoder

pd.set_option("display.max_columns", None)

params = {
    'dataset_root_dir': '../../sustainbench/data/dhs',
    'label': 'asset_index_normalized',
    'checkpoint_dir': './checkpoints', 
    'load_checkpoint': False,
    'batch_size': 512,
    'epochs': 50,
    'lr': 0.005,
    'num_rbf_anchor_pts': 13,
    'rbf_kernel_size':80,
    'model_type':"global",
    'device': 'cuda:0'
}

# Add file handler to save logs to a file
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join('logs', f'optuna_tuning_{params["label"]}_{current_time}.log')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

train_df = pd.read_csv(os.path.join(params['dataset_root_dir'], "dhs_trainval_labels.csv"))
val_df = pd.read_csv(os.path.join(params['dataset_root_dir'], "dhs_val_labels.csv"))
test_df = pd.read_csv(os.path.join(params['dataset_root_dir'], "dhs_test_labels.csv"))

train_df = train_df.dropna(subset=['asset_index_normalized'])
val_df = val_df.dropna(subset=['asset_index_normalized'])
test_df = test_df.dropna(subset=['asset_index_normalized'])

class RSDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get the nl_mean directly from the dataframe
        nl_mean = self.dataframe.iloc[idx]["nl_mean"]
        label = self.dataframe.iloc[idx]["asset_index_normalized"]
        if pd.isna(label):
            return None

        return nl_mean, label

train_dataset = RSDataset(train_df)
val_dataset = RSDataset(val_df)
test_dataset = RSDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)



class MLP(nn.Module):
    def __init__(self, train_dataset, device, num_rbf_anchor_pts=params['num_rbf_anchor_pts'], rbf_kernel_size=params['rbf_kernel_size']):
        super(MLP, self).__init__()
        self.position_encoder = RBFSpatialRelationPositionEncoder(train_locs=train_dataset, num_rbf_anchor_pts=num_rbf_anchor_pts, rbf_kernel_size=rbf_kernel_size, device=device)
        self.model = nn.Sequential(
            nn.Linear(self.position_encoder.pos_enc_output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        loc_embed = self.position_encoder(x)
        return self.model(loc_embed)

nl_mean_list = []
for batch in train_loader:
    nl_mean_batch, _ = batch
    nl_mean_list.extend(nl_mean_batch.numpy()) 

### train_nl_mean_array is a tensor of shape (num_train, 1)
train_nl_mean_array = np.array(nl_mean_list).reshape(-1, 1)

# model = MLP(train_dataset=train_nl_mean_array, device=params['device'])
# model = model.to(params['device'])

# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=params['lr'])

# best_val_loss = float("inf")

# start_epoch = 0
# if params['load_checkpoint']:
#     model, optimizer, best_val_loss, start_epoch = load_checkpoint(model, optimizer, params['checkpoint_dir'])

# num_epochs = params['epochs']

def objective(trial):
    num_rbf_anchor_pts = trial.suggest_int('num_rbf_anchor_pts', 1, 30)
    rbf_kernel_size = trial.suggest_int('rbf_kernel_size', 2, 50)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)

    model = MLP(train_dataset=train_nl_mean_array, device=params['device'], 
                num_rbf_anchor_pts=num_rbf_anchor_pts, rbf_kernel_size=rbf_kernel_size)
    model = model.to(params['device'])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_epochs = 30  # Use fewer epochs for tuning
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (nl_means, labels) in enumerate(train_loader):
            nl_means, labels = nl_means.to(params['device']), labels.to(params['device']).float()
            optimizer.zero_grad()
            nl_means = nl_means.reshape(nl_means.size(0), 1, 1)
            outputs = model(nl_means.cpu().numpy())
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * nl_means.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for nl_means, labels in val_loader:
                nl_means, labels = nl_means.to(params['device']), labels.to(params['device']).float()
                nl_means = nl_means.reshape(nl_means.size(0), 1, 1)
                outputs = model(nl_means.cpu().numpy())
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item() * nl_means.size(0)

        val_loss /= len(val_loader.dataset)

        logger.info(f"Trial {trial.number}, Epoch {epoch+1}/{num_epochs}, "
                    f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Calculate R² score on the test set
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for nl_means, labels in test_loader:
            nl_means, labels = nl_means.to(params['device']), labels.to(params['device']).float()
            nl_means = nl_means.reshape(nl_means.size(0), 1, 1)
            outputs = model(nl_means.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    r2 = r2_score(all_labels, all_preds)

    logger.info(f"Trial {trial.number}, Test R²: {r2:.4f}")

    return r2  

pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
study = optuna.create_study(direction='maximize', pruner=pruner)
study.optimize(objective, n_trials=500, timeout=7200) 

logger.info("Number of finished trials: %d", len(study.trials))
logger.info("Best trial:")

trial = study.best_trial

logger.info("  Value: %f", trial.value)
logger.info("  Params: ")
for key, value in trial.params.items():
    logger.info("    %s: %s", key, value)

# for epoch in range(start_epoch, num_epochs):
#     model.train()
#     train_loss = 0.0
#     for batch_idx, (nl_means, labels) in enumerate(train_loader):
#         nl_means, labels = nl_means.to(params['device']), labels.to(params['device']).float()
#         optimizer.zero_grad()
#         # nl_means shape is (batch_size, 1, 1)
#         nl_means = nl_means.reshape(nl_means.size(0), 1, 1)
#         outputs = model(nl_means.cpu().numpy())
#         loss = criterion(outputs.squeeze(), labels)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item() * nl_means.size(0)

#         if batch_idx % 10 == 0:
#             print(f"Epoch {epoch+1}/{params['epochs']}, Batch {batch_idx+1}/{len(train_loader)}, Training Loss: {loss.item():.4f}")

#     train_loss /= len(train_loader.dataset)

#     model.eval()
#     val_loss = 0.0
#     all_preds = []
#     all_labels = []
#     with torch.no_grad():
#         for nl_means, labels in test_loader:
#             nl_means, labels = nl_means.to(params['device']), labels.to(params['device']).float()
#             nl_means = nl_means.reshape(nl_means.size(0), 1, 1)
#             outputs = model(nl_means.cpu().numpy())
#             loss = criterion(outputs.squeeze(), labels)
#             val_loss += loss.item() * nl_means.size(0)
#             all_preds.append(outputs.cpu().numpy())
#             all_labels.append(labels.cpu().numpy())

#     val_loss /= len(test_loader.dataset)

#     all_preds = np.concatenate(all_preds)
#     all_labels = np.concatenate(all_labels)

#     r2 = r2_score(all_labels, all_preds)

#     print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}, Test Loss: {val_loss}, R²: {r2:.4f}")

#     is_best = val_loss < best_val_loss
#     best_val_loss = min(val_loss, best_val_loss)
    
#     save_checkpoint({
#         'epoch': epoch + 1,
#         'state_dict': model.state_dict(),
#         'best_val_loss': best_val_loss,
#         'optimizer': optimizer.state_dict(),
#     }, is_best, params['checkpoint_dir'])

# model.eval()
# test_loss = 0.0
# all_preds = []
# all_labels = []
# with torch.no_grad():
#     for batch_idx, (inputs, labels) in enumerate(test_loader):
#         inputs, labels = inputs.to(params['device']), labels.to(params['device']).float().unsqueeze(1)
#         inputs = inputs.reshape(inputs.size(0), 1, 1)
#         outputs = model(inputs.cpu().numpy())
#         loss = criterion(outputs, labels)
#         test_loss += loss.item() * inputs.size(0)
#         all_preds.append(outputs.cpu().numpy())
#         all_labels.append(labels.cpu().numpy())

#         if batch_idx % 10 == 0:
#             print(f"Test Batch {batch_idx+1}/{len(test_loader)}, Loss: {loss.item():.4f}")

# test_loss /= len(test_loader.dataset)
# all_preds = np.concatenate(all_preds)
# all_labels = np.concatenate(all_labels)

# r2 = r2_score(all_labels, all_preds)
# print(f"Test Loss: {test_loss:.4f}, R²: {r2:.4f}")
