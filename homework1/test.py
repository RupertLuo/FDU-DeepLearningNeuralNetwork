from train import Model
from dataloader import DataLoader
from tqdm import tqdm
import wandb
from loguru import logger
best_hyper_parameter = {
    'hidden_dim1':512,
    'lr':0.001,
    'weight_decay':0.00001,}
loader = DataLoader('./data')
train_img,train_label,test_img,test_label = loader.load_data()
model = Model(input_dim=784,hidden_dim1 = best_hyper_parameter['hidden_dim1'],out_dim = 10,weight_decay = best_hyper_parameter['weight_decay'],lr = best_hyper_parameter['lr'],test_img = test_img,test_label = test_label)
model.load_model()
model.eval()