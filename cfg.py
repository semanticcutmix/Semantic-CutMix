import os
from numpy.core.fromnumeric import resize
from model import *
from easydict import EasyDict
import torch.optim as optim
import torchvision.transforms as transforms
from augmentation import *
Cfg = EasyDict()

Cfg.use_gpu = '1' # 0 or 1


# Cfg.code_name = 'MobileNetV2_SmoothMix1_'
# Cfg.save_path = 'MobileNetV2_cifar10'
# Cfg.dataset = 'cifar10' # 'cifar10', 'cifar100' and 'imagenet' 
# Cfg.num_class = 10 # 10, 100, 1000
# Cfg.network = MobileNetV2(num_classes=Cfg.num_class)
# Cfg.image_size = 32
# Cfg.total_epoch = 250
# Cfg.batch_size = 128
# Cfg.mean =  [0.4914, 0.4822, 0.4465]
# Cfg.std =   [0.247, 0.243, 0.261]

######################################################################
# Cfg.code_name = 'resnet20_AugmentedSmoothMask1_'
# Cfg.save_path = 'resnet20_cifar10'
# Cfg.dataset = 'cifar10' # 'cifar10', 'cifar100' and 'imagenet' 
# Cfg.num_class = 10 # 10, 100, 1000
# Cfg.network = resnet(depth = 20, num_classes=Cfg.num_class)
# Cfg.image_size = 32
# Cfg.total_epoch = 250
# Cfg.batch_size = 128
# Cfg.mean =  [0.4914, 0.4822, 0.4465]
# Cfg.std =   [0.247, 0.243, 0.261]
######################################################################
Cfg.code_name = 'EfficientNetB0_SmoothMix1_'
Cfg.save_path = 'EfficientNetB0_cifar100'
Cfg.dataset = 'cifar100' # 'cifar10', 'cifar100' and 'imagenet' 
Cfg.num_class = 100 # 10, 100, 1000
Cfg.network = EfficientNetB0(num_classes=Cfg.num_class)
Cfg.image_size = 32
Cfg.total_epoch = 250
Cfg.batch_size = 128
Cfg.mean = [0.5070, 0.4865, 0.4409]
Cfg.std =  [0.2673, 0.2564, 0.2761]
######################################################################
# Cfg.code_name = 'resnet18_FMix1_'
# Cfg.save_path = 'resnet18_imagenet'
# Cfg.dataset = 'imagenet' # 'cifar10', 'cifar100' and 'imagenet' 
# Cfg.num_class = 1000 # 10, 100, 1000
# Cfg.network = torchvision.models.resnet18(pretrained = False)
# Cfg.image_size = 224
# Cfg.total_epoch = 20
# Cfg.batch_size = 128
# Cfg.mean = [0.485, 0.456, 0.406]
# Cfg.std =  [0.229, 0.224, 0.225]
######################################################################

Cfg.load_model = None
# Cfg.load_model = './MobileNetV2_cifar10/weight/MobileNetV2_SmoothMix1_3.pth'
Cfg.start_epoch = 0
# Cfg.restart_lr = 6.7821e-2
Cfg.learning_rate = 1e-1
Cfg.T_max = Cfg.total_epoch
Cfg.min_lr = 1e-8
Cfg.num_workers = 16
Cfg.momentum = 0.9
Cfg.weight_decay = 1e-4
Cfg.last_epoch = Cfg.start_epoch - 1

# Cfg.use_fm = True
Cfg.use_fm = False
Cfg.criterion = nn.CrossEntropyLoss()
Cfg.optimizer = optim.SGD

# Cfg.use_mp = True
Cfg.use_mp = False

# Cfg.data_mixing = noMix
# Cfg.data_mixing = MixUp
# Cfg.data_mixing = CutMix
Cfg.data_mixing = SmoothMix
# Cfg.data_mixing = AAM_withoutWCW
# Cfg.data_mixing = AAM_CAM
# Cfg.data_mixing = AAM_withoutWCW_mp
# Cfg.data_mixing = AAM_CAM_mp
# Cfg.data_mixing = AAM_batch_ft
# Cfg.data_mixing = AugmentedSmoothMask
# Cfg.data_mixing = AAM_batch_ft_camWeight

# Cfg.cam_mask = torch.load('./CAM_MAP_FROM_resnet20_GPARTWCWBM1_4.pt')
# Cfg.cam_mask = torch.load('CAM_MAP_FROM_resnet20_AAM_batch_ft_withoutWCW3_7.pt')
# Cfg.cam_mask = torch.load('CAM_MAP_FROM_resnet20_AAM_batch_ft_withoutWCW3_7_new.pt')

# Cfg.enable_CAM = True
Cfg.enable_CAM = False
if Cfg.enable_CAM:
    Cfg.cam_net = resnet(depth = 20, num_classes=10)
    PATH = 'resnet20_cifar10/weight/resnet20_AAM_batch_ft_camWeight1_5.pth'
    Cfg.cam_net.load_state_dict(torch.load(PATH))
    Cfg.cam_net.eval()
    Cfg.target_layer = Cfg.cam_net.layer3[2].conv2

Cfg.train_list = [3]




Cfg.training_transform = [   transforms.Compose([#0
                                transforms.ToTensor(),
                                transforms.Normalize(mean=Cfg.mean, std=Cfg.std),
                            ]),
                            transforms.Compose([#1
                                transforms.ToTensor(),
                                transforms.Normalize(mean=Cfg.mean, std=Cfg.std),
                            ]),
                            transforms.Compose([#2
                                transforms.ToTensor(),
                                # Cutout(0.6),
                                transforms.Normalize(mean=Cfg.mean, std=Cfg.std),
                            ]),
                            transforms.Compose([#3
                                transforms.ToTensor(),
                                # Cutout(0.8),
                                transforms.Normalize(mean=Cfg.mean, std=Cfg.std),
                            ]),
                            transforms.Compose([#4
                                transforms.ToTensor(),
                                # Cutout(1.0),
                                transforms.Normalize(mean=Cfg.mean, std=Cfg.std),
                            ]),
                            transforms.Compose([#5
                                transforms.RandomCrop(Cfg.image_size, padding=Cfg.image_size//8),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomAffine(degrees = (-20, 20)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=Cfg.mean, std=Cfg.std),
                            ]),
                            transforms.Compose([#6
                                transforms.RandomCrop(Cfg.image_size, padding=Cfg.image_size//8),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomAffine(degrees = (-20, 20)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=Cfg.mean, std=Cfg.std),
                            ]),
                            transforms.Compose([#7
                                transforms.RandomCrop(Cfg.image_size, padding=Cfg.image_size//8),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomAffine(degrees = (-20, 20)),
                                transforms.ToTensor(),
                                # Cutout(0.6),
                                transforms.Normalize(mean=Cfg.mean, std=Cfg.std),
                            ]),
                            transforms.Compose([#8
                                transforms.RandomCrop(Cfg.image_size, padding=Cfg.image_size//8),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomAffine(degrees = (-20, 20)),
                                transforms.ToTensor(),
                                # Cutout(0.8),
                                transforms.Normalize(mean=Cfg.mean, std=Cfg.std),
                            ]),
                            transforms.Compose([#9
                                transforms.RandomCrop(Cfg.image_size, padding=Cfg.image_size//8),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomAffine(degrees = (-20, 20)),
                                transforms.ToTensor(),
                                # Cutout(1.0),
                                transforms.Normalize(mean=Cfg.mean, std=Cfg.std),
                            ]),
                        ]

Cfg.validation_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=Cfg.mean, std=Cfg.std),
])


if not os.path.exists(Cfg.save_path):
    os.makedirs(Cfg.save_path)
if not os.path.exists(Cfg.save_path+ '/result'):
    os.makedirs(Cfg.save_path + '/result')
if not os.path.exists(Cfg.save_path + '/training-history'):
    os.makedirs(Cfg.save_path + '/training-history')
if not os.path.exists(Cfg.save_path + '/weight'):
    os.makedirs(Cfg.save_path + '/weight')



