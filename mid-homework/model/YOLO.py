
import torch, sys
sys.path.insert(0,'./')
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import OrderedDict
from torch.autograd import Variable
from configs.config_task2 import get_cfg_defaults

def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))
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
    def __init__(self):
        super(DarkNet, self).__init__()
        # conv1 32, 3*3        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        # batch norm
        self.bn1 = nn.BatchNorm2d(32)
        # leakyrelu
        self.relu1 = nn.LeakyReLU(0.1)
        """Residual Block"""
        # residual 1 ---- (32 + 64) * 1
        self.layer1 = self.residual_layer([32, 64], 1)
        # residual 2 ---- (64 + 128) * 2
        self.layer2 = self.residual_layer([64, 128], 2)
        # residual 3 ---- (128 + 256) * 8
        self.layer3 = self.residual_layer([128, 256], 8)
        # residual 4 ---- (256 + 512) * 8
        self.layer4 = self.residual_layer([256, 512], 8)
        # residual 5 ---- (512 + 1024) * 4
        self.layer5 = self.residual_layer([512, 1024], 4)

    def residual_layer(self, channel, blocks):
        layers = []
        layers.append(("ds_conv",nn.Conv2d(channel[0], channel[1], kernel_size=3, stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(channel[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))
        # residual block
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), BasicBlock(channel[1], channel)))
        return nn.Sequential(OrderedDict(layers))
        # 这是一个有顺序的容器，将特定神经网络模块按照在传入构造器的顺序依次被添加到计算图中执行

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_third = self.layer3(x)
        x_second = self.layer4(x_third)
        x_first = self.layer5(x_second)

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
        super(make_last_layers, self).__init__()
        self.branch1 = nn.Sequential(
            conv2d(input_channel, channel_list[0], 1),        
            conv2d(channel_list[0], channel_list[1], 3),    
            conv2d(channel_list[1], channel_list[0], 1),    
            conv2d(channel_list[0], channel_list[1], 3),
            conv2d(channel_list[1], channel_list[0], 1))
        self.branch2 = nn.Sequential(
            conv2d(channel_list[0], channel_list[1], 3),
            nn.Conv2d(channel_list[1], output_channel, kernel_size=1, stride=1, padding=0, bias=True)
        )
        
    def forward(self, x):
        out_branch = self.branch1(x) # 这一部分是流入下一个尺寸的concat的
        output = self.branch2(out_branch)
        return out_branch, output
    
class YoloBody(nn.Module):
    def __init__(self, cfg):
        super(YoloBody, self).__init__()
        self.cfg = cfg
        self.backbone = DarkNet()   # 获取darknet的结构

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
        out_first_branch, out_first  = self.last_layer_first(x_first)
        x_second_in = self.last_layer_second_conv(out_first_branch)
        x_second_in = self.last_layer_second_upsample(x_second_in)
        # 第一次concate 26,26,256 + 26,26,512 -> 26,26,768   
        x_second_in = torch.cat([x_second_in, x_second], 1)

        """第二个输出层 out_second = (batch_size,75,26,26)"""
        out_second_branch, out_second  = self.last_layer_second(x_second_in)
        x_third_in = self.last_layer_third_conv(out_second_branch)
        x_third_in = self.last_layer_third_upsample(x_third_in)
        # 第二次concate 52,52,128 + 52,525,256 ---> 52,52,384 # 堆叠过程
        x_third_in = torch.cat([x_third_in, x_third], 1)

        """第三个特征层 out_third = (batch_size ,75,52,52)"""
        _, out_third = self.last_layer_third(x_third_in)  # 最后一次只是需要其中的第一个预测这一部分
        return out_first, out_second, out_third
    
    def load_pretrained_model(self):
        self.backbone.load_state_dict(torch.load(self.cfg.MODEL['pretrained_path']))
        print("successful load pretrained model")


class Yolov3(nn.Module):
    def __init__(self, cfg):
        super(Yolov3, self).__init__()

        self.cfg = cfg

        self.body = YoloBody(cfg)

    def forward(self, x):
        out = []
        x_s, x_m, x_l = self.body(x)
        
        out.append(x_s)
        out.append(x_m)
        out.append(x_l)
        
        return out  # large, medium, small
    
    def load_pretrained_model(self):
        self.body.load_pretrained_model()
        
if __name__ == '__main__':
    cfg = get_cfg_defaults()
    cfg['training'] = True
    net = Yolov3(cfg)
    print(net)

    in_img = torch.randn(12, 3, 416, 416)
    p, p_d = net(in_img)

    for i in range(3):
        print(p[i].shape)
        print(p_d[i].shape)