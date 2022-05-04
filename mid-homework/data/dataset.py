from torchvision import datasets, transforms
import sys
sys.path.insert(0,'./')
from data.argument_type import Cutout
from configs.config import get_cfg_defaults
def load_dataset(cfg):
    # Image Preprocessing


    train_transform = transforms.Compose([])
    if cfg.DATA.augmentation:
        train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
    train_transform.transforms.append(transforms.ToTensor())
    test_transform = transforms.Compose([
        transforms.ToTensor()])
    if cfg.DATA.name[:5:] == 'cifar':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        train_transform.transforms.append(normalize)
        test_transform.transforms.append(normalize)
    if cfg.DATA.cutout:
        train_transform.transforms.append(Cutout(n_holes=cfg.CUTOUT.n_holes, length=cfg.CUTOUT.length))
    
    if cfg.DATA.name == 'cifar-10':
        num_classes = 10
        train_dataset = datasets.CIFAR10(root='/remote-home/rpluo/lrp_project/FDU-DeepLearningNeuralNetwork/mid-homework/dataset/',
                                        train=True,
                                        transform=train_transform,
                                        download=True)

        test_dataset = datasets.CIFAR10(root='/remote-home/rpluo/lrp_project/FDU-DeepLearningNeuralNetwork/mid-homework/dataset/',
                                        train=False,
                                        transform=test_transform,
                                        download=True)
    elif cfg.DATA.name == 'cifar-100':
        num_classes = 100
        train_dataset = datasets.CIFAR100(root='/remote-home/rpluo/lrp_project/FDU-DeepLearningNeuralNetwork/mid-homework/dataset/',
                                        train=True,
                                        transform=train_transform,
                                        download=True)

        test_dataset = datasets.CIFAR100(root='/remote-home/rpluo/lrp_project/FDU-DeepLearningNeuralNetwork/mid-homework/dataset/',
                                        train=False,
                                        transform=test_transform,
                                        download=True)
    elif cfg.DATA.name == 'VOC':
        num_classes = 20
        train_dataset = datasets.VOCDetection(root='./dataset/', year='2007', image_set='trainval', download=True, transform=train_transform)
        test_dataset = datasets.VOCDetection(root='./dataset/', year='2007', image_set='test', download=True, transform=test_transform)
    
    return train_dataset,test_dataset,num_classes

if __name__ == '__main__':
    cfg = get_cfg_defaults()
    cfg.merge_from_file("./configs/expriments.yaml")
    train_dataset,test_dataset,num_classes = load_dataset(cfg)
    for i in range(len(train_dataset)):
        img = train_dataset[i][0]
        print(1)
        
