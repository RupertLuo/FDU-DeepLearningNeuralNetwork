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
from data.argument_type import Mixup,Cutmix
from data.dataset import load_dataset
from torchvision import datasets, transforms
from model.resnet import ResNet18
import torch.nn as nn
from tqdm import tqdm
import wandb
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
def test(loader,model,device):
    model.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            pred = model(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    model.train()
    return val_acc
def main(cfg):
    wandb.init(project="mid-homework-task1", entity="rupert_luo")
    device = cfg.TRAIN.device
    train_dataset,test_dataset,num_classes = load_dataset(cfg)
    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=cfg.TRAIN.batch_size,
                                            shuffle=True,
                                            pin_memory=False,
                                            num_workers=2)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=cfg.TRAIN.batch_size,
                                            shuffle=False,
                                            pin_memory=False,
                                            num_workers=2)

    if cfg.MODEL.name == 'resnet18':
        model = ResNet18(num_classes=num_classes)
    

    model = model.to(device)

    # init mixup class for train
    if cfg.DATA.mixup:
        mixup = Mixup(cfg.MIXUP.alpha)
    if cfg.DATA.cutmix:
        cutmix = Cutmix(cfg.CUTMIX.cutmix_prob,cfg.CUTMIX.beta)
    

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.lr,
                                    momentum=0.9, nesterov=True, weight_decay=5e-4)
    best_acc = 0
    for epoch in range(cfg.TRAIN.epochs):
        xentropy_loss_avg = 0.
        correct = 0.
        total = 0.
        progress_bar = tqdm(train_loader)
        for i, (batch_x,batch_y) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))

            batch_x,batch_y = batch_x.to(cfg.TRAIN.device),batch_y.to(cfg.TRAIN.device)

            if cfg.DATA.mixup:
                batch_x, batch_y, lam= mixup.mixup_data(batch_x,batch_y)
            if cfg.DATA.cutmix:
                batch_x, batch_y, lam= cutmix.cutmix_data(batch_x,batch_y)

            pred = model(batch_x)

            if cfg.DATA.mixup:
                xentropy_loss= mixup.mixup_criterion(criterion, pred, batch_y, lam)
            elif cfg.DATA.cutmix:
                xentropy_loss= cutmix.cutmix_criterion(criterion, pred, batch_y, lam)
            else:
                xentropy_loss = criterion(pred, batch_y)


            optimizer.zero_grad()
            xentropy_loss.backward()
            optimizer.step()

            xentropy_loss_avg += xentropy_loss.item()
            pred = torch.max(pred.data, 1)[1]
            total += batch_y.size(0)
            correct += (pred == batch_y.data).sum().item()
            accuracy = correct / total
            progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)
            wandb.log({'loss':xentropy_loss.item()})
        test_acc = test(test_loader,model,device)
        tqdm.write('test_acc: %.3f' % (test_acc))

        row = {'epoch': epoch, 'train_acc': accuracy, 'test_acc': test_acc}
        if accuracy>best_acc:
            logger.info('save the model!!')
            torch.save(model.state_dict(),  Path(cfg.MODEL.saved_path)/cfg.MODEL.name/'best_model.pt')
            best_acc=accuracy
        logger.info(row)
        wandb.log(row)
    
    

            

    



if __name__ == "__main__":
    cfg = prepare_config()
    main(cfg)