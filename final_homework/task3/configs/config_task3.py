from yacs.config import CfgNode as CN
_C = CN()
_C.RANDOM_SEED = 44
_C.LOG_PATH = 'mid-homework/logs/train.log'
# DATA config
_C.DATA = CN()
_C.DATA.name = 'cifar-100'
_C.DATA.augmentation = False
_C.DATA.cutout = False
_C.DATA.cutmix = False
_C.DATA.mixup = False
_C.DATA.img_size = 224

# cutout config
_C.CUTOUT = CN()
_C.CUTOUT.n_holes = 1
_C.CUTOUT.length = 16

# mixup config
_C.MIXUP = CN()
_C.MIXUP.alpha = 1

# cutmix config
_C.CUTMIX = CN()
_C.CUTMIX.cutmix_prob = 0.5
_C.CUTMIX.beta = 1


# MODEL config
_C.MODEL = CN()
_C.MODEL.name = 'ViT'
_C.MODEL.saved_path = './FinalPJ/trained_model/'
_C.MODEL.classifier = 'token'
_C.MODEL.representation_size = None
_C.MODEL.hidden_size = 512 #768
_C.MODEL.patches = (16, 16)

_C.MODEL.transformer = CN()
_C.MODEL.transformer.attention_dropout_rate = 0.1
_C.MODEL.transformer.num_heads = 12
_C.MODEL.transformer.mlp_dim = 64
_C.MODEL.transformer.num_layers = 12
_C.MODEL.transformer.attention_dropout_rate = 0.0
_C.MODEL.transformer.dropout_rate = 0.1


# train config
_C.TRAIN = CN()
_C.TRAIN.num_workers = 4
_C.TRAIN.optimizer = 'adam'
_C.TRAIN.batch_size = 4
_C.TRAIN.lr = 5e-4
_C.TRAIN.weight_decay = 1e-5
_C.TRAIN.epochs = 10000
_C.TRAIN.warm_epochs = 500
_C.TRAIN.device = 'cuda:0'

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()



