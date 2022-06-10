# packge import 
import torch
from configs.config_task3 import get_cfg_defaults
from data.argument_type import Mixup,Cutmix
from data.dataset import load_cifar_dataset
from task3_train import prepare_config
from model.ViT import VisionTransformer
from torchmetrics.functional import accuracy, precision_recall,f1_score
from tqdm import tqdm

cfg = get_cfg_defaults()
cfg.TRAIN.batch_size = 3

train_dataset,test_dataset,num_classes = load_cifar_dataset(cfg)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=cfg.TRAIN.batch_size,
                                            shuffle=False,
                                            pin_memory=False,
                                            num_workers=2)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=cfg.TRAIN.batch_size,
                                            shuffle=False,
                                            pin_memory=False,
                                            num_workers=2)

model = VisionTransformer(cfg, zero_head=True, num_classes=100)

# baseline
'''
model.load_state_dict(torch.load('final_homework/task3/trained_model/ViT_(False, False, False, True)/best_model.pt'))
model = model.to('cuda:0')
'''
# cutout
model.load_state_dict(torch.load('final_homework/task3/trained_model/ViT_(False, False, False, True)/best_model.pt'))
model = model.to('cuda:0')
'''
#Cutmix
model.load_state_dict(torch.load('trained_model/resnet18_(False, False, True, False)/best_model.pt'))
model = model.to('cuda:0')
'''
# inference 
model.eval()
# train 
preds = []
targets = []
for images, labels in tqdm(train_loader):
    images = images.to('cuda:0')
    labels = labels.to('cuda:0')

    with torch.no_grad():
        pred = model(images)

    pred = torch.max(pred.data, 1)[1]
    preds.append(pred)
    targets.append(labels)
preds = torch.cat(preds)
targets = torch.cat(targets)
acc = accuracy(preds,targets)
pre,recall = precision_recall(preds, targets, average='macro', num_classes=num_classes)
f1 = f1_score(preds,targets,num_classes = num_classes)
print(acc.item(),pre.item(),recall.item(),f1.item())

# test
preds = []
targets = []
for images, labels in tqdm(test_loader):
    images = images.to('cuda:0')
    labels = labels.to('cuda:0')

    with torch.no_grad():
        pred = model(images)

    pred = torch.max(pred.data, 1)[1]
    preds.append(pred)
    targets.append(labels)
preds = torch.cat(preds)
targets = torch.cat(targets)
acc = accuracy(preds,targets)
pre,recall = precision_recall(preds, targets, average='macro', num_classes=num_classes)
f1 = f1_score(preds,targets,num_classes = num_classes)
print(acc.item(),pre.item(),recall.item(),f1.item())