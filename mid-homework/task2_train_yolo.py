import sys, os, random
import torch
sys.path.insert(0,'./')
from data.argument_type import Cutout
from configs.config_task2 import get_cfg_defaults
from data.dataset import VocCustomDataset, create_train_loader, create_valid_loader
from model.YOLO import Yolov3, init_weights
from utils import data_utils
from utils.yolo_loss import YOLOLoss
import torch.optim as optim
import numpy as np
from loguru import logger
from utils.data_utils import get_train_transform,get_valid_transform
from pathlib import Path
from tqdm import tqdm
import time
import wandb
def prepare_config():
    # set seeds
    cfg = get_cfg_defaults()
    cfg.merge_from_file(".//configs/experiments_task2.yaml")
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
    wandb.init(project="mid-homework-task2", entity="Guardian_zc")
    cfg = get_cfg_defaults()
    
    train_dataset = VocCustomDataset(cfg,'trainval', get_train_transform())
    valid_dataset = VocCustomDataset(cfg,'test',get_valid_transform())
    
    train_loader = create_train_loader(cfg,train_dataset)
    valid_loader = create_valid_loader(cfg,valid_dataset)

    logger.info(f"Number of training samples: {len(train_dataset)}")
    logger.info(f"Number of validation samples: {len(valid_dataset)}\n")
    model = Yolov3(cfg).to(cfg.TRAIN.device)
    
    init_weights(model)
    model.load_pretrained_model()
    
    
    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    # define the optimizer
    if cfg.TRAIN.optimizer =='SGD':
        optimizer = torch.optim.SGD(params, lr=cfg.TRAIN.lr, momentum=0.9, weight_decay=cfg.TRAIN.weight_decay)
    else:
        optimizer = torch.optim.Adam(params, lr=cfg.TRAIN.lr, weight_decay=cfg.TRAIN.weight_decay)
    min_loss = float('inf')
    loss_list = []
    
    criterion = YOLOLoss(anchors=cfg.MODEL["ANCHORS"], num_classes = len(cfg.DATA["classes"]), input_shape = [cfg.DATA["width"], cfg.DATA["height"]], cuda = True, anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    
    
    for epoch in range(cfg.TRAIN.epochs):
        logger.info(f"\nEPOCH {epoch+1} of {cfg.TRAIN.epochs}")

        prog_bar = tqdm(train_loader, total=len(train_loader))
        
        for i, data in enumerate(prog_bar):
            optimizer.zero_grad()
            images, targets = data
            boxes_list = list()
            images_list = []
            for img in images:
                images_list.append(img.unsqueeze(0))
            images_list = torch.cat(images_list)
            for target in targets:
                box = target['boxes']  
                box[:, [0, 2]] = box[:, [0, 2]] / cfg.DATA["width"]
                box[:, [1, 3]] = box[:, [1, 3]] / cfg.DATA["height"]

                box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
                box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
                labels = target['labels'].unsqueeze(1) 
                boxes = torch.cat((box, labels), 1).to(cfg.TRAIN.device)
                boxes_list.append(boxes)
            images_list = images_list.to(cfg.TRAIN.device)

            out = model(images_list) 
            loss_value_all = 0
            loss_all = 0
            
            for l in range(len(out)):
                loss_item = criterion(l, out[l], boxes_list)
                loss_value_all  += loss_item
                loss_all += loss_item.item()

            loss_list.append(loss_value_all.item())
            optimizer.zero_grad()
            loss_value_all.backward()
            optimizer.step()
            # update the loss value beside the progress bar for each iteration
            wandb.log({"train_loss":loss_all})
            prog_bar.set_description(desc=f"Loss: {loss_all:.4f}")
        # save the best model till now if we have the least loss in the...
        if sum(loss_list)/len(loss_list) < min_loss:
            logger.info("save_model!!")
            torch.save(model.state_dict(),  Path(cfg.MODEL.saved_path)/cfg.MODEL.name/'best_model.pt')
            min_loss = sum(loss_list)/len(loss_list)
        
if __name__ == "__main__":
    cfg = prepare_config()
    main(cfg)