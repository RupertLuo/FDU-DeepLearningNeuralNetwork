from model.resnet import ResNet18
from data.argument_type import Mixup,Cutmix
from data.dataset import load_cifar_dataset


ResNet = ResNet18()
total = sum([param.nelement() for param in ResNet.parameters()])
print("Number of ResNet 18 parameters: ",total)

