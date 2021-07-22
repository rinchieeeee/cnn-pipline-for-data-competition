import torch
import time
from torch.cuda.amp import autocast, GradScaler
from logger import AverageMeter, timeSince
import yaml

def train(train_loader, model, criterion, optimizer, epoch, scheduler, device, config):
    """
    config : config is config_general in trainer.py
    """
    #scaler = GradScaler()  # PyTorch のampライブラリを用いて, 高速化する. 32flopsを16に落として高速化.
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    
    model.train()
    start = end = time.time()
    
    for batch, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = data["image"].to(device)
        labels = data["label"].to(device)
        
        batch_size = labels.size(0)
        
        losses = train_step(images, labels, model, criterion, 
                            optimizer, scheduler, config, losses, batch)

        #経過時間の計算
        batch_time.update(time.time() - end)
        end = time.time()
        
        if batch % config["print_log_freq"] == 0 or batch == (len(train_loader) - 1):
            print("Epoch : [{0}][{1}/{2}]" 
                     "Data {data_time.val:.3f} ({data_time.avg: .3f}) "
                     "Elapsed {remain:s}"
                     "Loss : {loss.val: .4f}({loss.avg: .4f})".format(
                         epoch + 1, batch, len(train_loader), batch_time = batch_time, data_time = data_time, 
                         loss = losses, remain = timeSince(start, float(batch + 1)/len(train_loader))))
            
    return losses.avg


def train_step(images, labels, model, criterion, optimizer, scheduler, config, losses, batch):

    global_step = 0
    batch_size = labels.size(0)
    scaler = GradScaler()  # PyTorch のampライブラリを用いて, 高速化する. 32flopsを16に落として高速化.
    if config["fp16"] == True:
        
        with autocast():
            """
            torch.cuda.ampで計算するためのwithブロック
            """
            y_preds = model(images)
            #y_preds = y_preds.sigmoid()
            loss = criterion(y_preds, labels)  # calculate loss    

    
        # append loss history per batch
        losses.update(loss.item(), batch_size)
        if config["grad_acc_step"] > 1:
            loss = loss/config["grad_acc_step"]
        
        scaler.scale(loss).backward()
        
        """
        勾配が急激に増加して, 学習が不安定になることを防ぐために最大勾配ノルムを指定して, clippingする.
        CFG.max_grad_norm : default value is 1000.
        """
        #grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
    
        if (batch + 1) % config["grad_acc_step"] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1

    elif config["fp16"] == False:

        y_preds = model(images)
        loss = criterion(y_preds, labels)

        # append loss history per epoch
        losses.update(loss.item(), batch_size)
        if config["grad_acc_step"] > 1:
            loss = loss/config["grad_acc_step"]

        loss.backward()

        if (batch + 1) % config["grad_acc_step"] == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1


    return losses



def inference(valid_loader, model, criterion, device, config):
    """
    config : config_general in trainer.py

    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    
    model.eval()
    preds = []
    start = end = time.time()
    
    for batch, data in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = data["image"].to(device)
        labels = data["label"].to(device)
        batch_size = labels.size(0)
        
        # calculate loss
        with torch.no_grad():
            y_preds = model(images)
            
        #y_preds = y_preds.sigmoid()
        #loss = criterion(y_preds.sigmoid(), labels)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)
        # append accuracy history to preds list 
        preds.append(y_preds.to("cpu").numpy()) # convert numpy array to calulate accuracy using sklearn.
        #preds.append(y_presds.to("cpu").numpy())
        if config["grad_acc_step"] > 1:
            loss = loss/config["grad_acc_step"]
            
        # 経過時間を計算
        batch_time.update(time.time() - end)
        end = time.time()
        
        if batch % config["print_log_freq"] == 0 or batch == (len(valid_loader) - 1):
            print('EVAL: [{0}/{1}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(
                   batch, len(valid_loader), batch_time = batch_time,
                   data_time = data_time, loss = losses,
                   remain = timeSince(start, float(batch + 1)/len(valid_loader)),
                   ))
    predictions = np.concatenate(preds)
    return losses.avg, predictions




def train_tmp(train_loader, model, criterion, optimizer, epoch, scheduler, device, config):
    scaler = GradScaler()  # PyTorch のampライブラリを用いて, 高速化する. 32flopsを16に落として高速化.
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    
    model.train()
    start = end = time.time()
    global_step = 0
    
    for batch, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = data["image"].to(device)
        labels = data["label"].to(device)
        
        batch_size = labels.size(0)
        
        with autocast():
            """
            torch.cuda.ampで計算するためのwithブロック
            """
            y_preds = model(images)
            #y_preds = y_preds.sigmoid()
            loss = criterion(y_preds, labels)  # calculate loss    
            
        losses.update(loss.item(), batch_size)
        
        # append loss history per batch
        losses.update(loss.item(), batch_size)
        if config["grad_acc_step"] > 1:
            loss = loss/config["grad_acc_step"]
        
        scaler.scale(loss).backward()
        
        """
        勾配が急激に増加して, 学習が不安定になることを防ぐために最大勾配ノルムを指定して, clippingする.
        CFG.max_grad_norm : default value is 1000.
        """
        #grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        
        if (batch + 1) % config["grad_acc_step"] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
        
        losses = train_step(images, labels, model, criterion, 
                            optimizer, scheduler, config, losses)

        #経過時間の計算
        batch_time.update(time.time() - end)
        end = time.time()
        
        if batch % config["print_log_freq"] == 0 or batch == (len(train_loader) - 1):
            print("Epoch : [{0}][{1}/{2}]" 
                     "Data {data_time.val:.3f} ({data_time.avg: .3f}) "
                     "Elapsed {remain:s}"
                     "Loss : {loss.val: .4f}({loss.avg: .4f})".format(
                         epoch + 1, batch, len(train_loader), batch_time = batch_time, data_time = data_time, 
                         loss = losses, remain = timeSince(start, float(batch + 1)/len(train_loader))))
            
    return losses.avg