import sys
sys.path.append('final_homework/InverseForm')
import torch
from models.model_loader import load_model
from models.ocrnet import HRNet_Mscale
from utils.config import assert_and_infer_cfg
import logx
# laod model
logx.initialize(logdir='final_homework/task1/logs',
                    tensorboard=True,
                    global_rank=0)
model_path = 'final_homework/InverseForm/checkpoints/hrnet48_OCR_HMS_IF_checkpoint.pth'
arch = 'ocrnet.HRNet_Mscale'
result_path = None
num_classes = 19
assert_and_infer_cfg(None, 0, False, False, arch, '48', True, True)
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
net = HRNet_Mscale(num_classes, None, has_edge_head=False)
load_model(net, checkpoint)
# load image
# inference