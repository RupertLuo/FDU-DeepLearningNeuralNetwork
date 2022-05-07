import sys
sys.path.insert(0,'./')
import torch
import numpy as np
import cv2
import random
from configs.config_task2 import get_cfg_defaults
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

# define the training tranforms
def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        A.RandomRotate90(0.5),
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

# define the validation transforms
def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc', 
        'label_fields': ['labels']
    })
def draw_box(orig_image,draw_boxes,pred_classes,scores,CLASSES,COLORS,label = False):
    """
    [input]: origin_image: cv2 format image
             draw_boxes: a list contains boxes, like [[1,2,4,5],[3,4,7,8]]
             pred_classes: a list contains class_id of each box
             CLASSES: list of all Classname, 
             COLORS: rand_color_id list , COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    """
    for j, box in enumerate(draw_boxes):
        
        if label:
            class_name = pred_classes[j]
            class_score = scores[j]
            color = COLORS[CLASSES.index(class_name)]
            cv2.putText(orig_image, class_name, 
                        (int(box[0]), int(box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 
                        1, lineType=cv2.LINE_AA)
            cv2.putText(orig_image, str(class_score), 
                        (int(box[0]+50), int(box[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[5], 
                        1, lineType=cv2.LINE_AA)
        else:
            color = COLORS[5]
        
        cv2.rectangle(orig_image,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                color, 2)
