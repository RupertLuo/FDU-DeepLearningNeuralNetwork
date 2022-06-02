import sys
sys.path.append('final_homework/InverseForm')
import torch
from models.model_loader import load_model
from models.ocrnet import HRNet_Mscale,HRNet
from utils.config import assert_and_infer_cfg,cfg
# from runx.logx import logx
from pathlib import Path
from torchvision.datasets import ImageFolder
import torchvision.transforms as standard_transforms
import torchvision
from PIL import Image
from colorize import Colorize
import torch.nn.functional as F
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
import cv2
# laod model
# logx.initialize(logdir='final_homework/task1/logs',
#                     tensorboard=False,
#                     global_rank=0)
model_path = 'final_homework/InverseForm/checkpoints/hrnet48_OCR_HMS_IF_checkpoint.pth'
arch = 'ocrnet.HRNet_Mscale'
result_path = None
num_classes = 19
assert_and_infer_cfg(None, 0, False, False, arch, '48', True, True)
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
net = HRNet_Mscale(num_classes, None, has_edge_head=False)
load_model(net, checkpoint)
net.to('cuda:0')
# load image

vedio_img_dir = 'final_homework/task1/data/images'
vedio_img_list = sorted(list(Path(vedio_img_dir).glob('*.jpg')))
mean_std = (cfg.DATASET.MEAN, cfg.DATASET.STD)
val_input_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)
])
# inference
colorize = Colorize(19)
transform = A.Compose([
        ToTensorV2(p=1.0),
    ])
net.eval()
with torch.no_grad():
    for i,img_path in enumerate(tqdm(vedio_img_list)):
        image = cv2.imread(img_path)
        # convert BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = image_resized/255.0
        sample = transform(image=image_resized)
        image_resized = sample['image'].float()
        image = val_input_transform(image)
        image = image.unsqueeze(0)
        input_dict = {'images':image.cuda()}
        output = net(input_dict)
        pred = output['pred'][0]
        mask = F.softmax(pred, dim=1).max(0).indices
        color_pic = colorize(mask.cpu())
        pic= torchvision.transforms.functional.to_pil_image(color_pic)
        pic.save('final_homework/task1/data/image_seg/'+img_path.name)
