
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import OrderedDict
from torch.autograd import Variable
def init_weights(net, init_type='normal', gain=0.02):
    
    def init_func(m):
        classname = m.__class__.__name__

        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            # hasattr: Return whether the object has an attribute with the given name.

            if init_type == 'normal':
                init.normal_(m.weight.data, mean = 0.0, std = 0.01)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
            # specially initialize the parameters for batch normalization

    print('initialize network with %s' % init_type)
    net.apply(init_func)
    
class DarkNet(nn.Module):  # 主干DarkNet53
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        # conv1 32, 3*3        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        # batch norm
        self.bn1 = nn.BatchNorm2d(32)
        # leakyrelu
        self.relu1 = nn.LeakyReLU(0.1)
        """Residual Block"""
        # residual 1 ---- (32 + 64) * 1
        self.Residual_First = self.residual_layer([32, 64], 1)
        # residual 2 ---- (64 + 128) * 2
        self.Residual_Second = self.residual_layer([64, 128], 2)
        # residual 3 ---- (128 + 256) * 8
        self.Residual_Third = self.residual_layer([128, 256], 8)
        # residual 4 ---- (256 + 512) * 8
        self.Residual_Fouth = self.residual_layer([256, 512], 8)
        # residual 5 ---- (512 + 1024) * 4
        self.Residual_Fifth = self.residual_layer([512, 1024], 4)

    def residual_layer(self, channel, blocks):
        layers = []
        layers.append(nn.Conv2d(channel[0], channel[1], kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(channel[1]))
        layers.append(nn.LeakyReLU(0.1))
        # residual block
        for i in range(0, blocks):
            layers.append(BasicBlock(channel[1], channel))
        return nn.Sequential(OrderedDict(layers))
        # 这是一个有顺序的容器，将特定神经网络模块按照在传入构造器的顺序依次被添加到计算图中执行

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.Residual_First(x)
        x = self.Residual_Second(x)
        x_third = self.Residual_Third(x)
        x_second = self.Residual_Fouth(x_third)
        x_first = self.Residual_Fifth(x_second)

        return x_third, x_second, x_first

class BasicBlock(nn.Module):  
    def __init__(self, input_channel, channel):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, channel[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(channel[0])
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(channel[0], channel[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x   
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out += residual
        return out
    
class make_last_layers(nn.Module):
    def __init__(self, channel_list, input_channel, output_channel):
        # last convolutional layer
        self.branch1 = nn.ModuleList([
            nn.Conv2d(input_channel, channel_list[0], 1),        
            nn.Conv2d(channel_list[0], channel_list[1], 3),    
            nn.Conv2d(channel_list[1], channel_list[0], 1),    
            nn.Conv2d(channel_list[0], channel_list[1], 3),
            nn.Conv2d(channel_list[1], channel_list[0], 1)])
        self.branch2 = nn.ModuleList([
            nn.Conv2d(channel_list[0], channel_list[1], 3),
            nn.Conv2d(channel_list[1], output_channel, kernel_size=1, stride=1, padding=0, bias=True)
        ])
        
    def forward(self, x):
        out_branch = self.branch1(x) # 这一部分是流入下一个尺寸的concat的
        output = self.branch2(out_branch)
        return out_branch, output
    
class YoloBody(nn.Module):
    def __init__(self, cfg):
        super(YoloBody, self).__init__()
        self.cfg = cfg
        self.backbone = DarkNet(None)   # 获取darknet的结构

        # VOC 数据集共有 20 类, 因此最后输出的channel应该为 (20 + 4 + 1) * 3
        output_channel = (20 + 4 + 1) * 3
        self.last_layer_first = make_last_layers([512, 1024], 1024, output_channel)

        """第一个分支出来的数据进行一次1*1卷积加上采样,之后concat"""
        self.last_layer_second_conv = nn.Conv2d(512, 256, 1)    # 1*1 conv
        self.last_layer_second_upsample = nn.Upsample(scale_factor=2, mode='nearest')  # upsample
        self.last_layer_second = make_last_layers([256, 512], 512+256, output_channel)      # solution 2

        """第二个分支出来的数据进行一次1*1卷积加上采样,之后concat"""
        self.last_layer_third_conv = nn.Conv2d(256, 128, 1)   
        self.last_layer_third_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer_third = make_last_layers([128, 256], 256 + 128, output_channel)     # solution 3

    def forward(self, x):   
        #shape: 52*52*256   26*26*512    13*13*1024
        x_third, x_second, x_first = self.backbone(x)  # 将这个输入的x通过darknet提取三个特征层，用作堆叠。

        """第一个输出层 out_first = (batch_size,75,13,13)"""
        out_first, out_first_branch = self.last_layer_first(x_first)
        x_second_in = self.last_layer_second_conv(out_first_branch)
        x_second_in = self.last_layer_second_upsample(x_second_in)
        # 第一次concate 26,26,256 + 26,26,512 -> 26,26,768   
        x_second_in = torch.cat([x_second_in, x_second], 1)

        """第二个输出层 out_second = (batch_size,75,26,26)"""
        out_second, out_second_branch = self.last_layer_second(x_second_in)
        x_third_in = self.last_layer_two_conv(out_second_branch)
        x_third_in = self.last_layer_two_upsample(x_third_in)
        # 第二次concate 52,52,128 + 52,525,256 ---> 52,52,384 # 堆叠过程
        x_third_in = torch.cat([x_third_in, x_third], 1)

        """第三个特征层 out_third = (batch_size ,75,52,52)"""
        out_third, _ = self.last_layer_third(x_third_in)  # 最后一次只是需要其中的第一个预测这一部分
        return out_first, out_second, out_third
