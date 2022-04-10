import wandb
from loguru import logger
import itertools
from dataloader import DataLoader
from train import Model
config = {
    'hidden_dim1':[512,256,128],
    'lr':[0.005,0.001,0.0001],
    'weight_decay':[0.0001,0.00001],
}
experiment_settings = itertools.product(config['hidden_dim1'], config['lr'], config['weight_decay'])
loader = DataLoader('./data')
train_img,train_label,test_img,test_label = loader.load_data()
for setting in experiment_settings:
    run = wandb.init(project="deep_learning", reinit=True)
    wandb.run.name = 'hidden_dim1_%d_lr_%f_weight_decay_%f' % setting
    wandb.run.save()
    with run:
        logger.info('hidden_dim1: %d lr %f weight_decay %f' % setting)
        model = Model(input_dim=784,hidden_dim1 = setting[0],out_dim = 10,weight_decay = setting[2],lr = setting[1],test_img = test_img,test_label = test_label)
        model.train(train_img,train_label)
    


