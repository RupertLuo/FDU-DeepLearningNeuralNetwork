from yacs.config import CfgNode as CN
_C = CN()
_C.RANDOM_SEED = 44
# DATA config
_C.DATA = CN()
_C.DATA.name = 'cifar-100'
_C.DATA.augmentation = True
_C.DATA.cutout = True
_C.DATA.cutmix = False
_C.DATA.mixup = False
# cutout config
_C.CUTOUT = CN()
_C.CUTOUT.n_holes = 1
_C.CUTOUT.length = 16

# MODEL config
_C.MODEL = CN()
_C.MODEL.name = 'ResNet-18'



# train config
_C.TRAIN = CN()
_C.TRAIN.num_workers = 4
_C.TRAIN.optimizer = 'adam'
_C.TRAIN.batch_size = 4
_C.TRAIN.lr = 5e-4
_C.TRAIN.weight_decay = 1e-5
_C.TRAIN.epochs = 50
_C.TRAIN.device = 'cuda:0'

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()



