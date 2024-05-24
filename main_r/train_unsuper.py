import numpy as np
import matplotlib.pyplot as plt
import math
import os
import torch
import pickle
from argparse import ArgumentParser

from torch import optim
import models
import utils as ut
import datasets as dt
import data_utils as dtul
import grid_predictor as grid
from paths import get_paths
import losses as lo

from dataloader import *
from trainer_helper import *
from eval_helper import *
from trainer import *

parser = make_args_parser()
args = parser.parse_args()

trainer = Trainer(args, console=True)


trainer.run_train()
trainer.run_eval_final()
val_preds = trainer.run_eval_spa_enc_only(
    eval_flag_str="LocEnc ", load_model=True)
