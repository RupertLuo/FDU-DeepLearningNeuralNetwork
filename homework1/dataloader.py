from pathlib import Path
import numpy as np
import os, gzip, pickle
class DataLoader():
    def __init__(self,data_dir):
        self.data_dir = Path(data_dir)
        self.train_img = self.data_dir / 'train-images-idx3-ubyte.gz'
        self.train_label = self.data_dir / 'train-labels-idx1-ubyte.gz'
        self.test_img = self.data_dir / 't10k-images-idx3-ubyte.gz'
        self.test_label = self.data_dir / 't10k-labels-idx1-ubyte.gz'
        self.minist_data = self.data_dir / 'minist_data.pkl'
    def load_mnist(self,data_file, label_file):
        with gzip.open(data_file,'rb') as f:
            magic_num = int(f.read(4).hex(), 16) # 读取出的四个字节是16进制，需要转成10进制
            image_num = int(f.read(4).hex(), 16)
            image_width = int(f.read(4).hex(), 16)
            image_height = int(f.read(4).hex(), 16)
            img_data = np.frombuffer(f.read(), dtype='uint8') # 将剩余所有数据一次读取至numpy数组中
            img_data = img_data.reshape(image_num, image_width*image_height)
    
        with gzip.open(label_file,'rb') as f:
            magic_num = int(f.read(4).hex(), 16)
            label_num = int(f.read(4).hex(), 16)
            label_data = np.frombuffer(f.read(), dtype='uint8')
            label_data_matrix = np.zeros((label_num, 10))
            label_data_matrix = label_data_matrix.astype(np.uint8)
            for i in range(label_num):
                label_data_matrix[i, label_data[i]] = 1
        return img_data,label_data_matrix,label_data
    def load_data(self):
        if not os.path.exists(self.minist_data):
            train_img,train_label,_ = self.load_mnist(self.train_img,self.train_label)
            test_img,test_label,_ = self.load_mnist(self.test_img,self.test_label)
            with open(self.minist_data,'wb') as f:
                pickle.dump((train_img,train_label,test_img,test_label),f)
        else:
            with open(self.minist_data,'rb') as f:
                train_img,train_label,test_img,test_label = pickle.load(f)
        return train_img,train_label,test_img,test_label