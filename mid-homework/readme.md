# 期中作业

### 1. 使用CNN网络模型(自己设计或使用现有的CNN架构，如AlexNet，ResNet-18)作为baseline在CIFAR-100上训练并测试；对比cutmix, cutout, mixup三种方法以及baseline方法在CIFAR-100图像分类任务中的性能表现；对三张训练样本分别经过cutmix, cutout, mixup后进行可视化，一共show 9张图像。
    - 运行 task1_train.py 文件，cifa100数据集将被保存到 dataset/cifar-100-python 文件夹中，训好的resnet-18模型将被保存到 trained_model/resnet18_* 文件夹中。
    - 修改 configs/experiments_task1.yaml 配置文件将改变训练的数据增强方式(cutout,cutmix,mixup)。
    - 使用 task1_test.ipyb 中的脚本对几种增强方式进行测试，不同数据增强的模型表现在 task1_result.md 文件中
    - 各数据增强方法的图像可视化在 imgs/task1 文件夹中

### 2. 在VOC数据集上训练并测试目标检测模型Faster R-CNN和YOLO V3；在四张测试图像上可视化Faster R-CNN第一阶段的proposal box；两个训练好后的模型分别可视化三张不在VOC数据集内，但是包含有VOC中类别物体的图像的检测结果（类别标签，得分，boundingbox），并进行对比，一共show六张图像；
    - 在VOC2007数据集官方网站下载VOC2007数据集，并保存在dataset/VOCdevkit中
    - 运行 task2_train_fasterrcnn.py 和 task2_train_yoyo.py 在VOC2007数据集上训练faster_rcnn 和 yolov3模型，训好的模型将保存在trained_model中。
    - 运行 task2_inference_fasterrcnn.py, 并设置 46行 drawproposal= True, 能够可视化testset中图像在模型第一阶段输出的proposal_box，并保存在imgs/task2_output中
    - 分别运行 task2_inference_fasterrcnn.py 和 task2_inference_yolo.py, 能够可视化几张不在VOC2007数据集中的图像的物体检测结果，图像保存在 imgs/task3 中

各训练好的模型网盘地址为： https://drive.google.com/drive/folders/1S22IpvYl8s2Y9YMkccBXKSIJBd1prQDU?usp=sharing
