from model.resnet import ResNet18
from data.argument_type import Mixup,Cutmix
from data.dataset import load_cifar_dataset
from model.ViT import VisionTransformer
from pathlib import Path
from configs.config_task3 import get_cfg_defaults
from loguru import logger
import numpy as np
import torch
import wandb
import sys, random
import torch.nn as nn
from tqdm import tqdm
import math
from torch.optim.lr_scheduler import LambdaLR
ResNet = ResNet18()
total = sum([param.nelement() for param in ResNet.parameters()])

print("Number of ResNet 18 parameters: ",total)
# https://colab.research.google.com/drive/1h-RFjV6xqKwQhCBODGHzhmuRiiWtwXpX?usp=sharing#scrollTo=zMVosXYiZgap
class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

def prepare_config():
    # set seeds
    cfg = get_cfg_defaults()
    cfg.merge_from_file("FinalPJ\configs\experiments_task3.yaml")
    opts = [arg=='True' if i%2==1 else arg for i,arg in enumerate(sys.argv[1:])]
    cfg.merge_from_list(opts)
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
    model_output = Path(cfg.MODEL.saved_path)/(cfg.MODEL.name+"_"+str((cfg.DATA.augmentation,cfg.DATA.cutout,cfg.DATA.cutmix,cfg.DATA.mixup)))
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
    wandb.init(project="FinalPJ", entity="Guardian_zc")
    device = cfg.TRAIN.device
    train_dataset,test_dataset, _ = load_cifar_dataset(cfg)
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
    model = VisionTransformer(cfg, zero_head=True, num_classes=10)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of ViT parameters: ",total)

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
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=cfg.TRAIN.warm_epochs, t_total=cfg.TRAIN.epochs)
    model.zero_grad()
    
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

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scheduler.step()
            xentropy_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            xentropy_loss_avg += xentropy_loss.item()
            pred = torch.max(pred.data, 1)[1]
            if cfg.DATA.mixup or cfg.DATA.cutmix:
                if lam!= None:
                    rand_num = np.random.uniform()
                    if rand_num<lam:
                        batch_y = batch_y[0]
                    else:
                        batch_y = batch_y[1]
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
            torch.save(model.state_dict(),  Path(cfg.MODEL.saved_path)/(cfg.MODEL.name+"_"+str((cfg.DATA.augmentation,cfg.DATA.cutout,cfg.DATA.cutmix,cfg.DATA.mixup)))/'best_model.pt')
            best_acc=accuracy
        logger.info(row)
        wandb.log(row)
if __name__ == "__main__":
    cfg = prepare_config()
    main(cfg)
