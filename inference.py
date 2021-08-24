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

from tqdm.auto import tqdm
import sys
sys.path.append(f"{os.getcwd()}/src")

import torch
from torch.utils.data import DataLoader
from src.dataset import SetiTestDataset
from src.models import get_model
from src.transforms import get_transforms
import yaml
import argparse

from metrics import get_score
import ttach as tta
            

def seed_torch(seed = 0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



def make_submit(model, states, test_loader, device):

    model.to(device)
    probs = []
    
    for i, data in tqdm(enumerate(test_loader)):
        images = data["image"].to(device)
        avg_preds = []
        for state in states:
            model.load_state_dict(state['model'])
            model.eval()

            with torch.no_grad():
                y_preds = model(images.float())

            avg_preds.append(y_preds.sigmoid().to('cpu').numpy())

        # mean 5CV predict
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)
    probs = np.concatenate(probs)
    return probs


def make_submit_with_TTA(model, states, test_loader, device):

    transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            #tta.Rotate90(angles = [0, 90]),
        ]
    )

    tta_model = tta.ClassificationTTAWrapper(model, transforms, merge_mode = "mean")
    tta_model.to(device)
    probs = []
    
    for i, data in tqdm(enumerate(test_loader)):
        images = data["image"].to(device)
        avg_preds = []
        for state in states:
            tta_model.model.load_state_dict(state['model'])
            tta_model.eval()

            with torch.no_grad():
                y_preds = tta_model(images.float())

            avg_preds.append(y_preds.sigmoid().to('cpu').numpy())

        # mean 5CV predict
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)
    probs = np.concatenate(probs)
    return probs


def main(exp_file_name: str):

    """
    prepare something
    """

    with open (f"./config/{exp_file_name}.yml") as file:
        config = yaml.safe_load(file)
        
    config_general = config["general"]
    test = pd.read_csv(config_general["test_file"])

    if config_general["tta"]:
        print("Inplement Inference Mode: TTA")

    if config_general["debug"]:
        test = test.sample(n = 1000, random_state = config_general["seed"]).reset_index(drop = True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"We use {device}!")


    if not os.path.exists(config_general["output_dir"]):
        os.makedirs(config_general["output_dir"])

    seed_torch(seed = config_general["seed"])

    model = get_model(config)
    states = [torch.load(config_general["output_dir"] + "/" + config["model"]["name"] + f"_fold{fold}_best.pth") for fold in config_general["trn_fold"] ]
    
    test_dataset = SetiTestDataset(test, config, augmentation = get_transforms(config, "valid"))
    test_loader = DataLoader(test_dataset, 
                            batch_size = config["test"]["batch_size"], 
                            shuffle = False, 
                            num_workers = config_general["num_works"], 
                            pin_memory = True, 
                            drop_last = False)

    if config_general["tta"]:
        predictions = make_submit_with_TTA(model, states, test_loader, device)
        # submission
        test[config["target_cols"]] = predictions
        test.to_csv(config_general["output_dir"] + "/" + "submission_with_tta.csv", index = False)

    else:
        predictions = make_submit(model, states, test_loader, device)
        # submission
        test[config["target_cols"]] = predictions
        test.to_csv(config_general["output_dir"] + "/" + "submission.csv", index = False)

    print(test.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type = str, 
                        help = "your config file number e.g. your file is named exp001.yml, then you should set exp001")

    args = parser.parse_args()

    main(args.config)
