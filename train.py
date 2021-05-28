import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision
import torch.optim as optim
import PIL
from tqdm import tqdm
from cfg import Cfg
from utils import mixup_cross_entropy_loss, record_training_history
from augmentation import *
import copy
from torch.utils.data.sampler import SubsetRandomSampler
from dataset import *
from cam import CAM, GradCAM, GradCAMpp, SmoothGradCAMpp
from model import *
import torch.multiprocessing as mp

if Cfg.use_fm:
    fm = FMix(size = (Cfg.image_size, Cfg.image_size))

def train_and_val_fn(net, epoch, train, loader, criterion, optimizer, device, index):
    t = tqdm(loader, file=sys.stdout)
    if train:
        t.set_description('Epoch %i %s' % (epoch, "Training"))
        net.train()
    else:
        t.set_description('Epoch %i %s' % (epoch, "Validation"))
        net.eval()

    running_loss = 0.0
    total_loss = 0.0
    correct = 0
    total = 0
    
    pool = mp.Pool(Cfg.num_workers)

    for i, data in enumerate(loader, 0):

        
        #inputs, labels = data[0].to(device), data[1].to(device)
        inputs, labels = data[0], data[1]
        if train:
            with torch.no_grad():
                if Cfg.use_fm:
                    inputs = fm(inputs)
                else:
                    if Cfg.use_mp:
                        inputs, labels = Cfg.data_mixing(inputs, labels, (index%5+1)*2/10, Cfg, pool)
                    else:
                        inputs, labels = Cfg.data_mixing(inputs, labels, (index%5+1)*2/10, Cfg)
            optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)

        if train:
            if Cfg.use_fm:
                loss = fm.loss(outputs, labels)
            else:
                loss = mixup_cross_entropy_loss(outputs, labels)
            #loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        else:
            loss = criterion(outputs, labels)
        total_loss += loss.item()

        if not train:
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        t.set_postfix(loss = total_loss/(i+1))
        t.update()

    t.close()

    if train:
        average_loss = float(total_loss/len(loader))
        # acc = 100 * correct / total
        return average_loss#, acc
    else:
        average_loss = float(total_loss/len(loader))
        acc = 100 * correct / total
        return average_loss, acc

if __name__ == '__main__':
    
    if Cfg.use_mp:
        mp.set_start_method('spawn')

    device = torch.device(("cuda:" + Cfg.use_gpu) if torch.cuda.is_available() else "cpu")
    print("Training using", device)
    Cfg.device = device

    if Cfg.enable_CAM:
        Cfg.cam_model = CAM(Cfg.cam_net, Cfg.target_layer, device)
    else:
        cam_model = None
    
    print('This will train the augmentation set:', Cfg.train_list)

    valloader = get_val_dataloader(Cfg.dataset, Cfg.batch_size, Cfg.num_workers, transform = Cfg.validation_transform)

    for index in Cfg.train_list:

        print('Loading Dataset', Cfg.dataset)

        trainloader = get_train_dataloader(Cfg.dataset, Cfg.batch_size, Cfg.num_workers, transform = Cfg.training_transform[index])

        net = (copy.deepcopy(Cfg.network)).to(device)

        if Cfg.load_model is not None:
            net.load_state_dict(torch.load(Cfg.load_model))
            print('Loaded Model', Cfg.load_model)

        criterion = Cfg.criterion
        if Cfg.start_epoch == 0:
            optimizer = Cfg.optimizer(net.parameters(), lr=Cfg.learning_rate,
                            momentum=Cfg.momentum, weight_decay=Cfg.weight_decay)
        else:
            optimizer = Cfg.optimizer([{'params': net.parameters(), 'initial_lr': Cfg.learning_rate}], lr=Cfg.restart_lr,
                            momentum=Cfg.momentum, weight_decay=Cfg.weight_decay)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, last_epoch = Cfg.last_epoch, T_max= Cfg.T_max, eta_min = Cfg.min_lr, verbose=True)

        path_name = Cfg.code_name + str(index)

        best_acc = 0.0

        for epoch in range(Cfg.start_epoch, Cfg.total_epoch):

            print("-"*50)
            print("Training", path_name)

            average_train_loss = train_and_val_fn(net = net, epoch = epoch, train = True, loader = trainloader, criterion=criterion, optimizer=optimizer, device = device, index = index)
            
            with torch.no_grad():
                average_val_loss, val_acc = train_and_val_fn(net = net, epoch = epoch, train = False, loader = valloader, criterion=criterion, optimizer=optimizer, device = device, index = index)

            print("Average Training Loss :", average_train_loss)
            print("Average Test Loss :", average_val_loss,  "Validation acc :", val_acc, "%")
            
            scheduler.step()

            record_training_history(path = Cfg.save_path + '/training-history/' + path_name + '.txt', epoch = epoch, average_train_loss = average_train_loss, average_val_loss = average_val_loss)
            
            if best_acc < val_acc:
                best_acc = val_acc
                PATH = Cfg.save_path + '/weight/' + path_name + '.pth'
                torch.save(net.state_dict(), PATH)
                print("saved weight to", PATH)
            
            print("Best Acc :", best_acc)

        print("END")

        del trainloader, net, \
            criterion, optimizer, scheduler, path_name, \
            best_acc, epoch, average_train_loss, \
            average_val_loss, val_acc, PATH
