# 深度学习与神经网络 Homework1
## 下载数据
在MNIST数据集链接里（http://yann.lecun.com/exdb/mnist/）下载好数据文件，并放入./data文件夹中
## 训练
在命令行中输入以下命令，可以对模型进行训练，最好的模型将会保存在./model文件夹里
```
$ python train.py
```
## 超参数搜索
在文件 hyper_parameter.py 中的 config 变量中填入想要搜索的超参数，然后运行以下命令进行超参搜索
```
$ python hyper_parameter.py
```
## 测试以及权重可视化
运行以下命令对保存好的最好模型进行测试集测试，权重可视化图片将保存在./img文件夹里
```
$ python test.py
```
