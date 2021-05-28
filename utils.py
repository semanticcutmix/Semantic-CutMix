import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from time import sleep
import scipy.stats as st
import math
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import io
import torchvision
import imageio
def mixup_cross_entropy_loss(input, target, size_average=True):


    """Origin: https://github.com/moskomule/mixup.pytorch
    in PyTorch's cross entropy, targets are expected to be labels
    so to predict probabilities this loss is needed
    suppose q is the target and p is the input
    loss(p, q) = -\sum_i q_i \log p_i
    """
    assert input.size() == target.size()
    assert isinstance(input, Variable) and isinstance(target, Variable)
    input = torch.log(torch.nn.functional.softmax(input, dim=1).clamp(1e-5, 1))
    # input = input - torch.log(torch.sum(torch.exp(input), dim=1)).view(-1, 1)
    loss = - torch.sum(input * target)
    return loss / input.size()[0] if size_average else loss

def record_training_history(path, epoch, average_train_loss, average_val_loss):
    loss_history = open(path,"a")
    text = str(epoch) + " " + str(average_train_loss) + " " +  str(average_val_loss) + "\n"
    loss_history.writelines(text)
    loss_history.close()

def eval_net(net, testloader, device):

    net.eval()
    correct = 0
    total = 0

    iterrt = iter(testloader)

    while True:
        try:
            images, label = next(iterrt)

            label = label.to(device)

            outs = net(images.to(device))

            _, predicted = torch.max(outs, 1)
            correct += (predicted == label).sum().item()

            total += label.size(0)

        except StopIteration:
            break
    return 100 * correct / total

def record_result(PATH, index, acc):
    loss_history = open(PATH,"a")
    text = str(index) + " " + str(acc) + "\n"
    loss_history.writelines(text)
    loss_history.close()

def reverse_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    x[:, 0, :, :] = x[:, 0, :, :] * std[0] + mean[0]
    x[:, 1, :, :] = x[:, 1, :, :] * std[1] + mean[1]
    x[:, 2, :, :] = x[:, 2, :, :] * std[2] + mean[2]
    return x


def visualize(img, cam):
    """
    Synthesize an image with CAM to make a result image.
    Args:
        img: (Tensor) shape => (1, 3, H, W)
        cam: (Tensor) shape => (1, 1, H', W')
    Return:
        synthesized image (Tensor): shape =>(1, 3, H, W)
    """

    _, _, H, W = img.shape
    cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
    cam = 255 * cam.squeeze()
    heatmap = cv2.applyColorMap(np.uint8(cam), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap.transpose(2, 0, 1))
    heatmap = heatmap.float() / 255
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])

    result = heatmap + img.cpu()
    result = result.div(result.max())

    return result

def visualize_big(img, cam):
    """
    Synthesize an image with CAM to make a result image.
    Args:
        img: (Tensor) shape => (1, 3, H, W)
        cam: (Tensor) shape => (1, 1, H', W')
    Return:
        synthesized image (Tensor): shape =>(1, 3, H, W)
    """

    _, _, H, W = img.shape
    cam = F.interpolate(cam, size=(720, 720), mode='bilinear', align_corners=False)
    cam = 255 * cam.squeeze()
    heatmap = cv2.applyColorMap(np.uint8(cam), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap.transpose(2, 0, 1))
    heatmap = heatmap.float() / 255
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])

    result = heatmap# + img.cpu()
    result = result.div(result.max())

    return result

def smooth_heatmap(cam, image_size = 32):
    H, W = image_size, image_size
    cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
    cam = 255 * cam.squeeze()
    heatmap = cv2.applyColorMap(np.uint8(cam), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap.transpose(2, 0, 1))
    heatmap = heatmap.float() / 255
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])

    return heatmap

def smooth_heatmap_single_color(cam, image_size = 32):
    H, W = image_size, image_size
    cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
    cam = 255 * cam.squeeze()
    return cam

def show_history(PATH_LIST):
    i = 0

    for PATH in PATH_LIST:
        angle_record = []
        acc_record = []

        history = open(PATH, "r")
        line = history.readline()
        while line:
            angle, acc = line.split()
            angle_record.append(int(angle))
            acc_record.append(float(acc))
            line = history.readline()
        label, _ = os.path.splitext(os.path.basename(PATH))
        plt.figure(0)
        plt.plot(angle_record, acc_record, label=label)
        plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(12,6)
    plt.xlabel("Angle")
    plt.ylabel("Accarucy")
    plt.show(block=True)


def show_training_history(PATH):
    epochs = []
    train_loss_record = []
    val_loss_record = []
    train_acc_record = []
    val_acc_record = []

    history = open(PATH, "r")
    line = history.readline()
    while line:
        epoch, train_loss, val_loss = line.split()
        epochs.append(int(epoch))
        train_loss_record.append(float(train_loss))
        val_loss_record.append(float(val_loss))
        line = history.readline()
    plt.figure(0)
    fig = plt.gcf()
    fig.set_size_inches(13, 5)
    plt.plot(epochs, train_loss_record, 'b', label="training loss")
    plt.plot(epochs, val_loss_record, 'r', label="validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show(block=True)

def gkern(kernlen, nsig, mag):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return (kern2d/kern2d.sum()) * mag

def gkern_torch(kernlen, nsig, mag):
    x_range = torch.arange(0, kernlen, 1)
    y_range = torch.arange(0, kernlen, 1)
    xx, yy = torch.meshgrid(x_range, y_range)
    pos = torch.empty(xx.shape + (2,))
    pos[:, :, 0] = xx
    pos[:, :, 1] = yy

    # var_torch = torch.tensor([nsig], dtype=torch.float32)
    # mag_torch = torch.tensor([mag], dtype=torch.float32)
    # mean_torch = torch.tensor([32,32])

    return ((1./(2.*math.pi*nsig)) * torch.exp(-torch.sum((pos - ([32,32]))**2., dim=-1) /(2*nsig))) * mag

def denormalize_imshow(img, path):
    img = img *0.2 + 0.5     # unnormalize
    npimg = img.numpy()
    img = np.transpose(npimg, (1, 2, 0))
    img[img > 1.0] = 1.0
    plt.imshow(img)
    if path is not None:
        plt.imsave(path, img)
    plt.show()

def denormalize_imshow_with_bar(img, path = None):
    img[img > 1.0] = 1.0
    plt.imshow(img[0,:,:])
    plt.colorbar()
    if path is not None:
        plt.savefig(path)
    plt.show()

def gaussian2d_torch(size, var):
    x_range = torch.arange(0, size, 1)
    y_range = torch.arange(0, size, 1)
    xx, yy = torch.meshgrid(x_range, y_range)
    pos = torch.empty(xx.shape + (2,))
    pos[:, :, 0] = xx
    pos[:, :, 1] = yy
    mean = size // 2

    return ( torch.exp(-((pos[:, :, 0] - mean) ** 2)/(2 * (var ** 2)))) * ( torch.exp(-((pos[:, :, 1] - mean) ** 2)/(2 * (var ** 2))))


def imageNetTrainDataset(root_dir, transform = None):
    # def target_transform(x):
    #     return x + 1
    return torchvision.datasets.ImageFolder(root = root_dir, transform = transform)

class imageNetValidDataset(Dataset):

    def __init__(self, label_file, root_dir, transform=None):
        self.labels = pd.read_csv(label_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = 'ILSVRC2012_val_'+ '%08d' % (idx + 1) + '.JPEG'
        img_name = os.path.join(self.root_dir, path)
        image = imageio.imread(img_name)
        label = self.labels['labels'][idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# class imageNetTestDataset(Dataset):

#     def __init__(self, label_file, root_dir, transform=None):
#         self.labels = pd.read_csv(label_file)
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         img_name = os.path.join(self.root_dir, idx)
#         image = io.imread(img_name)
#         label = self.labels[idx]

#         if self.transform:
#             image = self.transform(image)

#         return image, label




