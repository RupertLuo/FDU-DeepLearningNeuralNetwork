import torch
import cv2
import numpy as np
import os
import glob as glob
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from xml.etree import ElementTree as et
import sys
sys.path.insert(0,'mid-homework')
from data.argument_type import Cutout
from utils.data_utils import collate_fn
from configs.config_task1 import get_cfg_defaults as get_cfg_defaults1
from configs.config_task2 import get_cfg_defaults as get_cfg_defaults2
from tqdm import tqdm
def load_cifar_dataset(cfg):
    # Image Preprocessing
    train_transform = transforms.Compose([])
    if cfg.DATA.augmentation:
        train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
    train_transform.transforms.append(transforms.ToTensor())
    test_transform = transforms.Compose([
        transforms.ToTensor()])
    # if cfg.DATA.name[:5:] == 'cifar':
    #     normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    #                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    #     train_transform.transforms.append(normalize)
    #     test_transform.transforms.append(normalize)
    if cfg.DATA.cutout:
        train_transform.transforms.append(Cutout(n_holes=cfg.CUTOUT.n_holes, length=cfg.CUTOUT.length))
    
    if cfg.DATA.name == 'cifar-10':
        num_classes = 10
        train_dataset = datasets.CIFAR10(root='/remote-home/rpluo/lrp_project/FDU-DeepLearningNeuralNetwork/mid-homework/dataset/',
                                        train=True,
                                        transform=train_transform,
                                        download=True)

        test_dataset = datasets.CIFAR10(root='/remote-home/rpluo/lrp_project/FDU-DeepLearningNeuralNetwork/mid-homework/dataset/',
                                        train=False,
                                        transform=test_transform,
                                        download=True)
    elif cfg.DATA.name == 'cifar-100':
        num_classes = 100
        train_dataset = datasets.CIFAR100(root='/remote-home/rpluo/lrp_project/FDU-DeepLearningNeuralNetwork/mid-homework/dataset/',
                                        train=True,
                                        transform=train_transform,
                                        download=True)

        test_dataset = datasets.CIFAR100(root='/remote-home/rpluo/lrp_project/FDU-DeepLearningNeuralNetwork/mid-homework/dataset/',
                                        train=False,
                                        transform=test_transform,
                                        download=True)

    return train_dataset,test_dataset,num_classes

class VocCustomDataset(Dataset):
    def __init__(self,cfg, partition,transforms=None):
        self.transforms = transforms
        self.dir_path = Path(cfg.DATA.dir)
        self.height = cfg.DATA.height
        self.width = cfg.DATA.width
        self.classes = cfg.DATA.classes
        self.partition = partition

        
        # get all the image paths in sorted order
        self.all_img_id = sorted(map(lambda x:x.strip(),open(self.dir_path/'ImageSets'/'Main'/(self.partition+'.txt'),'r').readlines()))
        self.image_dir= self.dir_path/'JPEGImages'
        self.annotation_dir = self.dir_path/'Annotations'
        

    def __getitem__(self, idx):
        # capture the image name and the full image path
        image_name = self.all_img_id[idx]
        image_path = str(self.image_dir/(image_name+'.jpg'))

        # read the image
        image = cv2.imread(image_path)
        # convert BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        ori_h,ori_w,_ = image.shape
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized = image_resized/255.0
        
        # capture the corresponding XML file for getting the annotations
        annot_filename = image_name + '.xml'
        annot_file_path = str(self.annotation_dir/annot_filename)
        
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        
        # get the height and width of the image
        image_width = image.shape[1]
        image_height = image.shape[0]
        
        # box coordinates for xml files are extracted and corrected for image size given
        for member in root.findall('object'):
            # map the current object name to `classes` list to get...
            # ... the label index and append to `labels` list
            labels.append(self.classes.index(member.find('name').text))
            
            # xmin = left corner x-coordinates
            xmin = int(member.find('bndbox').find('xmin').text)
            # xmax = right corner x-coordinates
            xmax = int(member.find('bndbox').find('xmax').text)
            # ymin = left corner y-coordinates
            ymin = int(member.find('bndbox').find('ymin').text)
            # ymax = right corner y-coordinates
            ymax = int(member.find('bndbox').find('ymax').text)
            
            # resize the bounding boxes according to the...
            # ... desired `width`, `height`
            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            yamx_final = (ymax/image_height)*self.height
            
            boxes.append([xmin_final, ymin_final, xmax_final, yamx_final])
        
        # bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # no crowd instances
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        target["image_name"] = image_name
        target["origin_shape"] = (ori_w,ori_h)

        # apply the image transforms
        
        if self.transforms:
            sample = self.transforms(image = image_resized,
                                     bboxes = target['boxes'],
                                     labels = labels)
            image_resized = sample['image'].float()
            target['boxes'] = torch.Tensor(sample['bboxes'])
            
        return image_resized, target

    def __len__(self):
        return len(self.all_img_id)

def create_train_loader(cfg,train_dataset):
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.batch_size,
        shuffle=True,
        num_workers=cfg.TRAIN.num_workers,
        collate_fn=collate_fn
    )
    return train_loader
def create_valid_loader(cfg,valid_dataset):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.TRAIN.batch_size,
        shuffle=True,
        num_workers=cfg.TRAIN.num_workers,
        collate_fn=collate_fn
    )
    return valid_loader

if __name__ == '__main__':
    cfg = get_cfg_defaults2()
    cfg.merge_from_file("mid-homework/configs/experiments_task2.yaml")
    dataset = VocCustomDataset(cfg,'trainval')
    for i in tqdm(range(len(dataset))):
        data = dataset[0]