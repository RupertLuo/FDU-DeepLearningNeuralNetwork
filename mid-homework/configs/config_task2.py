from yacs.config import CfgNode as CN
import numpy as np
_C = CN()
_C.RANDOM_SEED = 44
_C.LOG_PATH = './logs/train.log'
# DATA config
_C.DATA = CN()
_C.DATA.name = 'VOC'
_C.DATA.dir = './dataset/VOCdevkit/VOC2007'
_C.DATA.width = 416
_C.DATA.height = 416
_C.DATA.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']
_C.DATA.num_classes = len(_C.DATA.classes)
# MODEL config
_C.MODEL = CN()
_C.MODEL.name = 'YOLOv3'
_C.MODEL.saved_path = './trained_model/'
_C.MODEL.pretrained_path = './trained_model/darknet53_backbone_weights.pth'
_C.MODEL.ANCHORS = [np.array([10.,13.]),  np.array([16.,30.]),  np.array([33.,23.]),  np.array([30.,61.]),  np.array([62.,45.]),  np.array([59.,119.]),  np.array([116.,90.]),  np.array([156.,198.]),  np.array([373.,326.])]
_C.MODEL.STRIDES = [8, 16, 32]
_C.MODEL.ANCHORS_PER_SCLAE = 3


# train config
_C.TRAIN = CN()
_C.TRAIN.num_workers = 4
_C.TRAIN.optimizer = 'adam'
_C.TRAIN.batch_size = 8
_C.TRAIN.lr = 5e-5
_C.TRAIN.weight_decay = 1e-5
_C.TRAIN.epochs = 50
_C.TRAIN.device = 'cuda:0'

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()