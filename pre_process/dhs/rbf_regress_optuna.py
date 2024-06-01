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
from utils import save_checkpoint, load_checkpoint, RBFFeaturePositionEncoder

pd.set_option("display.max_columns", None)

import argparse

# Define the command-line arguments
parser = argparse.ArgumentParser(description="Hyperparameter Optimization with Optuna")
parser.add_argument('--dataset_root_dir', type=str, default='../../sustainbench/data/dhs', help='Root directory of the dataset')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
parser.add_argument('--load_checkpoint', action='store_true', help='Load checkpoint if available')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
parser.add_argument('--num_rbf_anchor_pts', type=int, default=13, help='Number of RBF anchor points')
parser.add_argument('--rbf_kernel_size', type=int, default=80, help='RBF kernel size')
parser.add_argument('--model_type', type=str, default='global', help='Type of model')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
parser.add_argument('--label', type=str, default='experiment_1', help='Label for the experiment')

args = parser.parse_args()
params = {
    'dataset_root_dir': args.dataset_root_dir,
    'checkpoint_dir': args.checkpoint_dir,
    'load_checkpoint': args.load_checkpoint,
    'batch_size': args.batch_size,
    'epochs': args.epochs,
    'lr': args.lr,
    'num_rbf_anchor_pts': args.num_rbf_anchor_pts,
    'rbf_kernel_size': args.rbf_kernel_size,
    'model_type': args.model_type,
    'device': args.device,
    'label': args.label
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

train_df = train_df.dropna(subset=[params['label']])
val_df = val_df.dropna(subset=[params['label']])
test_df = test_df.dropna(subset=[params['label']])

class RSDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get the nl_mean directly from the dataframe
        nl_mean = self.dataframe.iloc[idx]["nl_mean"]
        label = self.dataframe.iloc[idx][params['label']]
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
    def __init__(self, train_dataset, device, num_rbf_anchor_pts, rbf_kernel_size, layers, neurons, act_func):
        super(MLP, self).__init__()
        self.position_encoder = RBFFeaturePositionEncoder(
            train_locs=train_dataset, 
            num_rbf_anchor_pts=num_rbf_anchor_pts, 
            rbf_kernel_size=rbf_kernel_size, 
            device=device
        )
        self.model = self._build_model(self.position_encoder.pos_enc_output_dim, layers, neurons, act_func)

    def _build_model(self, input_dim, layers, neurons, act_func):
        modules = []
        for i in range(layers):
            modules.append(nn.Linear(input_dim if i == 0 else neurons[i-1], neurons[i]))
            modules.append(act_func)
        modules.append(nn.Linear(neurons[-1], 1))
        return nn.Sequential(*modules)

    def forward(self, x):
        loc_embed = self.position_encoder(x)
        return self.model(loc_embed)

nl_mean_list = []
for batch in train_loader:
    nl_mean_batch, _ = batch
    nl_mean_list.extend(nl_mean_batch.numpy()) 

### train_nl_mean_array is a tensor of shape (num_train, 1)
train_nl_mean_array = np.array(nl_mean_list).reshape(-1, 1)

# Define the objective function for Optuna
def objective(trial):
    num_rbf_anchor_pts = trial.suggest_int('num_rbf_anchor_pts', 1, 30)
    rbf_kernel_size = trial.suggest_int('rbf_kernel_size', 2, 50)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)

    # Hyperparameters for the MLP
    layers = trial.suggest_int('layers', 1, 3)
    neurons = [trial.suggest_int(f'neurons_l{i}', 8, 128) for i in range(layers)]
    activation_choices = {'ReLU': nn.ReLU(), 'Tanh': nn.Tanh(), 'LeakyReLU': nn.LeakyReLU()}
    activation_name = trial.suggest_categorical('activation', list(activation_choices.keys()))
    activation = activation_choices[activation_name]

    model = MLP(train_dataset=train_nl_mean_array, device=params['device'], 
                num_rbf_anchor_pts=num_rbf_anchor_pts, rbf_kernel_size=rbf_kernel_size, 
                layers=layers, neurons=neurons, act_func=activation)
    model = model.to(params['device'])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_epochs = 1  # Use fewer epochs for tuning
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

pruner = MedianPruner(n_startup_trials=100, n_warmup_steps=10, interval_steps=5)
study = optuna.create_study(direction='maximize', pruner=pruner)
study.optimize(objective, n_trials=500, timeout=50400)  # 50400 seconds = 14 hours

logger.info("Number of finished trials: %d", len(study.trials))
logger.info("Best trial:")

trial = study.best_trial

logger.info("  Value: %f", trial.value)
logger.info("  Params: ")
for key, value in trial.params.items():
    logger.info("    %s: %s", key, value)
