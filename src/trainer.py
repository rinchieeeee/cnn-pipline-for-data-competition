import time
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from models import get_model
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from optimizer import get_optimizer
from train import train, inference
import yaml
from logger import *
import numpy as np

def trainer(folds: pd.DataFrame, fold: int, device, config : dict, LOGGER):
    
    LOGGER.info(f"=========== fold : {fold} training ============")
    
    config_train = config["train"]
    # loading data and split to OOF
    train_idx = folds[folds.fold != fold].index
    val_idx = folds[folds.fold == fold].index
    
    train_df = folds.loc[train_idx].reset_index(drop = True)
    val_df = folds.loc[val_idx].reset_index(drop = True)
    val_labels = val_df[config["target_cols"]].values
    
    train_dataset = CustomDataset(train_df, augmentation = get_transforms(data = "train"))
    valid_dataset = CustomDataset(val_df, augmentation = get_transforms(data = "valid"))
    
    # drop_last = Trueは, dataset sizeがbatchで割り切れなかった時に, 最後に残るbatchを捨てて, 学習しないことを意味する.
    # こうする事により, norm batch系などでエラーが出なくなるよ.
    train_loader = DataLoader(train_dataset, batch_size = config["train_batch_size"], 
                                shuffle = True, 
                                num_workers = config_train["num_workers"], 
                             pin_memory = True, drop_last = True)
    
    valid_loader = DataLoader(valid_dataset, batch_size = config_train["val_batch_size"], 
                             shuffle = False, num_workers = config_train["num_workers"], 
                             pin_memory = True, drop_last = False)
    
    """
    Scheduler
    """
    def get_scheduler(optimizer, config):
        config_scheduler = config["scheduler"]

        if config_scheduler["name"] == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(optimizer, mode = "min", 
                                          factor = CFG.factor, patience = CFG.patience, 
                                          verbose = True, eps = CFG.eps)
        
        elif config_scheduler["name"] == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(optimizer, **config_scheduler["params"])
            
        elif config_scheduler["name"] == "CosineAnnealingWarmRestarts":
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = CFG.T_0, 
                                                   T_mult = 1, eta_min = CFG.min_lr, last_epoch = -1)
        return scheduler
    
    
    
    """
    seting model and optimizer
    """
    model = get_model(config)
    model.to(device)
    
    optimizer = get_optimizer(model.parameters(), config)
    #optimizer = Adam(model.parameters(), lr = CFG.lr, weight_decay = CFG.weight_decay, amsgrad = False)
    scheduler = get_scheduler(optimizer)
    
    # epoch部分
    
    #criterion = nn.BCEWithLogitsLoss()
    #criterion = FocalLoss()  # (logit, taregt)
    criterion = get_lossfunc(config_train["loss_function"])
    best_score = 0.0
    best_loss = np.inf # for early stopping
    early_stopping_count = 0

    LOGGER = init_logger()
    
    for epoch in range(config_train["epochs"]):
        start_time = time.time()
        
        # train
        avg_loss = train(train_loader, model, criterion, optimizer, epoch, scheduler, device, config["general"])
        
        # eval
        avg_val_loss, preds = inference(valid_loader, model, criterion, device, config["general"])
        
        scheduler.step()
            
        # スコアを計算
        score, scores = get_score(val_labels, preds)
        
        elapsed = time.time() - start_time
        
        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}')

        
        if avg_val_loss < best_loss:
            early_stopping_count = 0
            best_loss = avg_val_loss
            LOGGER.info(f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
            torch.save({'model': model.state_dict(), 
                        'preds': preds},
                        config["general"]["output_dir"] + "/" + f'{config["general"]["name"]}_fold{fold}_best.pth')
            
        else:
            early_stopping_count += 1
            
        if early_stopping_count >= config_train["patience"]:
            break
    
    check_point = torch.load(config["general"]["output_dir"] + "/" + f'{config["model"]["name"]}_fold{fold}_best.pth')
    for c in [f'pred_{c}' for c in config["target_cols"]]:
        val_df[c] = np.nan
    val_df[[f'pred_{c}' for c in config["target_cols"]]] = check_point['preds']
    #val_df["pred_label"] = np.nan
    #val_df.pred_label = check_point["preds"]

    return val_df
