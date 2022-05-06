from configs.config_task2 import get_cfg_defaults
from model.faster_RCNN import create_model

from tqdm.auto import tqdm
from data.dataset import VocCustomDataset, create_train_loader, create_valid_loader
from utils.data_utils import get_train_transform,get_valid_transform

import torch
import random
import numpy as np
from loguru import logger
from pathlib import Path
import time
import wandb
def prepare_config():
    # set seeds
    cfg = get_cfg_defaults()
    cfg.merge_from_file("mid-homework/configs/experiments_task2.yaml")
    cfg.freeze()
    torch.manual_seed(cfg.RANDOM_SEED)
    random.seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    # make output dir
    logger.add(cfg.LOG_PATH,
        level='DEBUG',
        format='{time:YYYY-MM-DD HH:mm:ss} - {level} - {file} - {line} - {message}',
        rotation="10 MB")
    logger.info("Train config: %s" % str(cfg))
    model_output = Path(cfg.MODEL.saved_path)/(cfg.MODEL.name)
    model_output.mkdir(exist_ok =True)
    return cfg


    
def main(cfg):
    wandb.init(project="mid-homework-task2", entity="rupert_luo")
    train_dataset = VocCustomDataset(cfg,'trainval',get_train_transform())
    valid_dataset = VocCustomDataset(cfg,'test',get_valid_transform())
    train_loader = create_train_loader(cfg,train_dataset)
    valid_loader = create_valid_loader(cfg,valid_dataset)

    logger.info(f"Number of training samples: {len(train_dataset)}")
    logger.info(f"Number of validation samples: {len(valid_dataset)}\n")

    # initialize the model and move to the computation device
    model = create_model(num_classes=cfg.DATA.num_classes)
    model = model.to(cfg.TRAIN.device)
    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    # define the optimizer
    if cfg.TRAIN.optimizer =='SGD':
        optimizer = torch.optim.SGD(params, lr=cfg.TRAIN.lr, momentum=0.9, weight_decay=cfg.TRAIN.weight_decay)
    else:
        optimizer = torch.optim.Adam(params, lr=cfg.TRAIN.lr, weight_decay=cfg.TRAIN.weight_decay)
    min_loss = float('inf')
    loss_list = []
    # start the training epochs
    for epoch in range(cfg.TRAIN.epochs):
        logger.info(f"\nEPOCH {epoch+1} of {cfg.TRAIN.epochs}")

        prog_bar = tqdm(train_loader, total=len(train_loader))
        for i, data in enumerate(prog_bar):
            optimizer.zero_grad()
            images, targets = data
            
            images = list(image.to(cfg.TRAIN.device) for image in images)
            targets = [{k: v.to(cfg.TRAIN.device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            loss_list.append(loss_value)
            losses.backward()
            optimizer.step()

        
            # update the loss value beside the progress bar for each iteration
            wandb.log({"train_loss":loss_value})
            prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

        # save the best model till now if we have the least loss in the...
        logger.info(f"\nEPOCH loss: {sum(loss_list)/len(loss_list)}")
        if sum(loss_list)/len(loss_list) < min_loss:
            logger.info("save_model!!")
            torch.save(model.state_dict(),  Path(cfg.MODEL.saved_path)/cfg.MODEL.name/'best_model.pt')
            min_loss = sum(loss_list)/len(loss_list)
        

if __name__ == "__main__":
    cfg = prepare_config()
    main(cfg)