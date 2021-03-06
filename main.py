import warnings 
warnings.filterwarnings('ignore')

import os
import pandas as pd

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
from src.trainer import trainer
from src.logger import init_logger
import yaml
import argparse

from metrics import get_score
            

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

    LOGGER = init_logger(config_general)
    seed_torch(seed = config_general["seed"])

    folds = train.copy()

    Fold = get_cv_split(config)
    for n, (train_index, val_index) in enumerate(Fold.split(folds, folds.target)):
        folds.loc[val_index, 'fold'] = int(n)

    folds['fold'] = folds['fold'].astype(int)

    
    def get_result(result_df):
        preds = result_df[[f'pred_{c}' for c in config["target_cols"]]].values
        labels = result_df[config["target_cols"]].values
        score, scores = get_score(labels, preds)
        LOGGER.info(f'Score : {score: <.4f} Scores : {np.round(scores, decimals = 4)}')

        
    if config_general["train"]:
        oof_df = pd.DataFrame()
        for fold in range(config_general["n_fold"]):
            if fold in config_general["trn_fold"]:
                _oof_df = trainer(folds, fold, device, config, LOGGER)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"======== fold : {fold} result =========")
                get_result(_oof_df)
        
        # CV result
        LOGGER.info(f"========= CV ========")
        get_result(oof_df)
        oof_df.to_csv(config_general["output_dir"] + "/" + "oof_df.csv", index = False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type = str, 
                        help = "your config file number e.g. your file is named exp001.yml, then you should set exp001")

    args = parser.parse_args()

    main(args.config)
