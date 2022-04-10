from train import Model
from dataloader import DataLoader
from tqdm import tqdm
import wandb
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
best_hyper_parameter = {
    'hidden_dim1':512,
    'lr':0.001,
    'weight_decay':0.00001,}
loader = DataLoader('./data')
train_img,train_label,test_img,test_label = loader.load_data()
model = Model(input_dim=784,hidden_dim1 = best_hyper_parameter['hidden_dim1'],out_dim = 10,weight_decay = best_hyper_parameter['weight_decay'],lr = best_hyper_parameter['lr'],test_img = test_img,test_label = test_label)
model.load_model()
acc,loss = model.eval()
# 可视化矩阵权重
plt.matshow(model.params['W1'])
plt.savefig('./img/W1.jpg')
plt.matshow(model.params['W2'])
plt.savefig('./img/W2.jpg')