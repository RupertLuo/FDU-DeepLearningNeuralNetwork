from yacs.config import CfgNode as CN
_C = CN()
_C.RANDOM_SEED = 44
_C.LOG_PATH = 'mid-homework/logs/train.log'
# DATA config
_C.DATA = CN()
_C.DATA.name = 'VOC'
_C.DATA.dir = 'mid-homework/dataset/VOCdevkit/VOC2007'
_C.DATA.width = 416
_C.DATA.height = 416
_C.DATA.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']
_C.DATA.num_classes = len(_C.DATA.classes)

# MODEL config
_C.MODEL = CN()
_C.MODEL.name = 'faster_rcnn'
_C.MODEL.saved_path = 'mid-homework/trained_model/'



# train config
_C.TRAIN = CN()
_C.TRAIN.num_workers = 16
_C.TRAIN.optimizer = 'adam'
_C.TRAIN.batch_size = 16
_C.TRAIN.lr = 5e-4
_C.TRAIN.weight_decay = 1e-5
_C.TRAIN.epochs = 50
_C.TRAIN.device = 'cuda:6'

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()



