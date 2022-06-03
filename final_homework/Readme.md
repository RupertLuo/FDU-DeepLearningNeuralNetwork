1. 使用在Cityscapes数据集上开源的任意一个语义分割模型，网络下载一段驾驶视频（类似行车记录仪视频），对视频每一帧进行测试并可视化，结果视频上传至网盘；
2. 对Faster R-CNN模型，分别进行以下训练：a) 随机初始化训练VOC；b) ImageNet预训练backbone网络，然后使用VOC进行fine tune；c)使用coco训练的Mask R-CNN的backbone网络参数，初始化Faster R-CNN的backbone网络，然后使用VOC进行fine tune；
3. 设计与期中作业1模型相同参数量的Transformer网络模型，进行CIFAR-100的训练，并与期中作业1的模型结果进行比较，可使用data aug