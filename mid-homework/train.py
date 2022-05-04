from loguru import logger
from pip import main
import torch
import numpy as np
import random
from pathlib import Path
import wandb
from configs.config import get_cfg_defaults
import torchvision
import torchvision.transforms as transforms
from data.dataset import load_dataset
from torchvision import datasets, transforms
from model.resnet import ResNet18
import torch.nn as nn
def prepare_config():
    # set seeds
    cfg = get_cfg_defaults()
    cfg.merge_from_file("/remote-home/rpluo/lrp_project/FDU-DeepLearningNeuralNetwork/mid-homework/configs/expriments.yaml")
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
    model_output = Path(cfg.MODEL.saved_path)/cfg.MODEL.name
    model_output.mkdir(exist_ok =True)
    return cfg
def main(cfg):
    device = cfg.TRAIN.device
    train_dataset,test_dataset,num_classes = load_dataset(cfg)
    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=cfg.TRAIN.batch_size,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=2)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=cfg.TRAIN.batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=2)

    if cfg.MODEL.name == 'resnet18':
        cnn = ResNet18(num_classes=num_classes)
    

    cnn = cnn.to(device)
    criterion = nn.CrossEntropyLoss().cuda()
    cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=cfg.TRAIN.lr,
                                    momentum=0.9, nesterov=True, weight_decay=5e-4)

    



if __name__ == "__main__":
    cfg = prepare_config()
    main(cfg)