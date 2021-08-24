import warnings 
warnings.filterwarnings('ignore')

import os
import math
import time
import random
import shutil

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from tqdm.auto import tqdm
#from functools import partial
import sys
sys.path.append(f"{os.getcwd()}/src")

import torch 
from src.trainer_for_lr_find import trainer
from src.logger import init_logger
import yaml
import argparse

from metrics import get_score
import matplotlib.pyplot as plt
plt.style.use("ggplot")
            

def seed_torch(seed = 0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_cv_split(config):
    """
    config : config is original file with loading yaml
    """
    config_split = config["split"]

    if config_split["name"] == "StratifiedKFold":
        return StratifiedKFold(**config_split["params"])

def plot_lr_find(lr_init_list, each_lr_scores, config):
    config_plot = config["lr_range_test"]["plot"]
    plt.plot(lr_init_list, each_lr_scores, "-x")
    plt.xlabel("Learning Rate")
    plt.ylabel(config_plot["y_axis"])
    plt.savefig(config["general"]["output_dir"] + "/" + "lr_range_test.png")
    


def main(exp_file_name: str):

    """
    prepare something
    """

    with open (f"./config/{exp_file_name}.yml") as file:
        config = yaml.safe_load(file)
        
    config_general = config["general"]
    train = pd.read_csv(config_general["train_file"])
    #test = pd.read_csv("../input/ranzcr-clip-catheter-line-classification/sample_submission.csv")

    if config_general["debug"]:
        train = train.sample(n = 1000, random_state = config_general["seed"]).reset_index(drop = True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"We use {device}!")


    if not os.path.exists(config_general["output_dir"]):
        os.makedirs(config_general["output_dir"])

    #LOGGER = init_logger(config_general)
    seed_torch(seed = config_general["seed"])

    folds = train.copy()

    Fold = get_cv_split(config)
    for n, (train_index, val_index) in enumerate(Fold.split(folds, folds.target)):
        folds.loc[val_index, 'fold'] = int(n)

    folds['fold'] = folds['fold'].astype(int)

    each_lr_scores = []
    #lr_init_list = [1e-6, 3e-6, 5e-6, 1e-5, 3e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2]
    lr_init_list = [3e-6, 5e-6, 1e-5, 3e-5, 5e-5, 1e-4, 3e-4, 5e-4, 6e-4, 7e-4, 8e-4, 1e-3, 3e-3, 5e-3, 7e-3, 1e-2]
    for lr_init in lr_init_list:
        cv_scores = []
        for fold in range(config_general["n_fold"]):
            if fold in config["lr_range_test"]["trn_fold"]:
                cv_score = trainer(folds, fold, device, lr_init, config)
                cv_scores.append(cv_score)

        print(f"lr_init = {lr_init} then cv score is {np.mean(cv_scores)}")
        each_lr_scores.append(np.mean(cv_scores))


    plot_lr_find(lr_init_list, each_lr_scores, config)
        
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type = str, 
                        help = "your config file number e.g. your file is named exp001.yml, then you should set exp001")

    args = parser.parse_args()

    main(args.config)
