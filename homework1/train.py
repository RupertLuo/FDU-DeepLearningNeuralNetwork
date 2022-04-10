from unittest import loader
from pathlib import Path
import numpy as np
from dataloader import DataLoader
from tqdm import tqdm
import wandb
from loguru import logger
class Model():
    def __init__(self,input_dim,hidden_dim1,out_dim,weight_decay,lr,test_img ,test_label):
        self.weight_decay = weight_decay
        self.activate = self.ReLU
        self.params = {
        'W1':np.random.randn(hidden_dim1, input_dim) * np.sqrt(1. / hidden_dim1),
        'b1':np.random.randn(hidden_dim1, 1) * np.sqrt(1. / hidden_dim1),
        'W2':np.random.randn(out_dim, hidden_dim1) * np.sqrt(1. / out_dim),
        'b2':np.random.randn(out_dim, 1) * np.sqrt(1. / out_dim)
    }
        self.lr = lr
        self.save_path = Path('./model/')
        self.test_img = test_img
        self.test_label = test_label
    def ReLU(self,x):
        return np.maximum(0,x)
    def ReLu_d(self,x):
        return (x>=0).astype(np.float64)
    def softmax(self, x):
        exps = np.exp(x - x.max(0))
        return exps / np.sum(exps, axis=0)
    def softmax_d(self,x):
        exps = np.exp(x - x.max(0))
        return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
    def forward(self,x):
        params = self.params
        params['A0'] = x.T
        params['Z1'] = np.dot(params["W1"], params['A0']) + params['b1']
        params['A1'] = self.ReLU(params['Z1'])
        params['Z2'] = np.dot(params["W2"], params['A1'])+ params['b2']
        params['A3'] = self.softmax(params['Z2'])
        return params['A3']
    def backward(self,output,y_train,batch_size):
        params = self.params
        change_w = {}
        # Calculate W2 update
        error = np.sum(output.T - y_train,0)/batch_size# (10,1)
        change_w['W2'] = np.outer(error, np.sum(params['A1'],1)/batch_size)# 512 764 10x1 512 x1
        change_w['b2'] = error.reshape(-1,1)
        # Calculate W1 update
        error = np.dot(params['W2'].T, error) * self.ReLu_d(np.sum(params['Z1'],1)/batch_size)
        change_w['W1'] = np.outer(error, np.sum(params['A0'],1)/batch_size)
        change_w['b1'] = error.reshape(-1,1)
        # Calculate weight decay
        change_w['W1'] += self.weight_decay * params['W1']
        change_w['W2'] += self.weight_decay * params['W2']
        
        return change_w
    def cross_entropy(self,pre_y,y):
        predictions = np.clip(pre_y, 1e-12, 1.-1e-12).T
        N = predictions.shape[0]
        ce = - np.sum(y*np.log(predictions)) / N
        return ce
    def update_params(self,change_w):
        params = self.params
        for key in change_w:
            params[key] -= self.lr * change_w[key]
    def save_model(self):
        params = self.params
        np.save(self.save_path/'W1.npy',params['W1'])
        np.save(self.save_path/'W2.npy',params['W2'])
        np.save(self.save_path/'b1.npy',params['b1'])
        np.save(self.save_path/'b2.npy',params['b2'])
    def load_model(self):
        params = self.params
        params['W1'] = np.load(self.save_path/'W1.npy')
        params['W2'] = np.load(self.save_path/'W2.npy')
        params['b1'] = np.load(self.save_path/'b1.npy')
        params['b2'] = np.load(self.save_path/'b2.npy')

    def train(self,img,label):
        batch_size = 3
        epoch_size = 10
        for epoch in range(epoch_size):
            epoch_loss = 0
            cnt = 0
            best_acc = 0
            dataset_length = len(img)
            iteration_size = dataset_length // batch_size
            for k in range(iteration_size):
                # sample batch
                batch_index = np.random.choice(range(dataset_length), batch_size,replace=False)
                batch_img, batch_label = img[batch_index], label[batch_index]
                # do batch training
                output = self.forward(batch_img)
                loss = self.cross_entropy(output,batch_label)
                epoch_loss += loss
                change_w = self.backward(output,batch_label,batch_size)
                # accuracy count
                cnt+=np.sum(np.argmax(batch_label,1)==np.argmax(output,0))
                self.update_params(change_w)
                wandb.log({"batch_loss":loss})
            train_acc = cnt/dataset_length
            test_acc,test_loss = self.eval()
            wandb.log({"train_loss":epoch_loss/iteration_size,'test_loss':test_loss,'train_accurary':train_acc,'test_accurary':test_acc})
            logger.info('epoch: %d train_loss = %.4f test_loss = %.4f train_accurary = %.4f test_accurary = %.4f'%(epoch,epoch_loss/iteration_size,test_loss,train_acc,test_acc))
            # save models
            if test_acc>best_acc:
                best_acc = test_acc
                self.save_model()

    def eval(self):
        output = self.forward(self.test_img)
        pred = np.argmax(output,0)
        gt = np.argmax(self.test_label,1)
        acc = np.sum(pred==gt)/len(gt)
        loss = self.cross_entropy(output,self.test_label)
        return acc,loss


if __name__ == '__main__':
    wandb.init(project="deep_learning")
    loader = DataLoader('./data')
    train_img,train_label,test_img,test_label = loader.load_data()
    model = Model(input_dim=784,hidden_dim1 = 512,out_dim = 10,weight_decay = 0.00001,lr = 0.0001,test_img = test_img,test_label = test_label)
    model.train(train_img,train_label)
    