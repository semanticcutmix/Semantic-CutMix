from cam import get_CAM
import torch
import numpy as np
import random
import scipy.stats as st
from utils import *
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from torch import nn
from PIL import Image


class random_corruption(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.random() > self.p:
            return img

        input_image = img
        ctype = np.random.randint(5, size=1)
    
        input_image = np.array(input_image)
        if ctype == 0:
            input_image[8:24,22:28,:] = 0
        if ctype == 1:
            weight_mask = gkern(64,5,800)
            weight_mask = weight_mask[8:40,8:40]
            weight_mask = np.float32(weight_mask)
            weight_mask = np.stack([weight_mask,weight_mask,weight_mask], axis = 0)
            input_image -= weight_mask
        if ctype == 2:
            input_image[:,:,24:32] += 1.0
        if ctype == 3:
            weight_mask = gkern(64,4,700)
            weight_mask = weight_mask[4:36,4:36]
            weight_mask = np.float32(weight_mask)
            weight_mask = np.stack([weight_mask,weight_mask,weight_mask], axis = 0)
            input_image += weight_mask
        if ctype == 4:
            input_image[:,16:28,16:28] = 1 - input_image[:,0:12,0:12]
        img = torch.tensor(input_image)
        return img

################# Data Transform ################

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, p, n_holes = 1, length = 16):
        self.n_holes = n_holes
        self.length = length
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """

        if np.random.random() > self.p:
            return img
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def HideAndSeek(p):

    def _HideAndSeek(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image
        mask = torch.zeros((image.shape[0],image.shape[1]))
        step = image.shape[0] // 4
        for i in range(0,image.shape[0],step):
            for j in range(0,image.shape[1],step):
                if torch.rand(1) < 0.5:
                    image[i:i+step,j:j+step,:] = 0.0
        return image

    return _HideAndSeek

class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, p, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.p = p
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.p:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img

class GridMask(nn.Module):
    def __init__(self, p, use_h = True, use_w = True, rotate = 1, offset=False, ratio = 0.5, mode=1):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = p

        self.prob = p
    # def set_prob(self, epoch, max_epoch):
    #     self.prob = self.st_prob * epoch / max_epoch #+ 1.#0.5

    def forward(self, x):
        if np.random.rand() > self.prob or not self.training:
            return x
        #n,c,h,w = x.size()
        c,h,w = x.size()
        x = x.view(-1,h,w).cuda()
        hh = int(1.5*h)
        ww = int(1.5*w)
        d = np.random.randint(2, h)
        #d = self.d
        #self.l = int(d*self.ratio+0.5)
        self.l = min(max(int(d*self.ratio+0.5),1),d-1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh//d):
                s = d*i + st_h
                t = min(s+self.l, hh)
                mask[s:t,:] *= 0
        if self.use_w:
            for i in range(ww//d):
                s = d*i + st_w
                t = min(s+self.l, ww)
                mask[:,s:t] *= 0
       
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
#        mask = 1*(np.random.randint(0,3,[hh,ww])>0)
        mask = mask[(hh-h)//2:(hh-h)//2+h, (ww-w)//2:(ww-w)//2+w]

        mask = torch.from_numpy(mask).float().cuda()
        if self.mode == 1:
            mask = 1-mask
        mask = mask.expand_as(x).cuda()
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h,w) - 0.5)).float().cuda()
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask 

        return x.view(c,h,w)
        #return x.view(n,c,h,w)

################## Data Mixing ##################

def noMix(imgs, labels, p, Cfg):

    two_hot_labels = torch.zeros((imgs.shape[0],Cfg.num_class)).to(Cfg.device)

    for n in range(imgs.shape[0]-1):
        two_hot_labels[n,int(labels[n])] = 1.0

    two_hot_labels[imgs.shape[0]-1,int(labels[imgs.shape[0]-1])] = 1.0

    return imgs, two_hot_labels


def MixUp(imgs, labels, p, Cfg):
    two_hot_labels = torch.zeros((imgs.shape[0],Cfg.num_class)).to(Cfg.device)

    for n in range(imgs.shape[0]-1):
        if torch.rand(1) < p:

            lam = np.random.beta(1.0, 1.0)

            original_image = imgs[n,:,:,:].to(Cfg.device)
            patch_image = imgs[n+1,:,:,:].to(Cfg.device)

            mask = torch.zeros((32,32)).to(Cfg.device)
            mask[:,:] = lam

            patch_mask = mask.clone().detach().to(Cfg.device)
            patch_mask_inv = 1 - patch_mask.clone().detach().to(Cfg.device)

            new_image = torch.zeros_like(imgs[n,:,:,:]).to(Cfg.device)

            # patch_mask = TF.rotate(patch_mask, int(angle))

            new_image[:] += patch_image[:] * patch_mask
            new_image[:] += original_image[:] * patch_mask_inv
            imgs[n] = new_image

            
            ratio = (torch.sum(patch_mask))/(32*32)
            ratio2 = (torch.sum(patch_mask_inv))/(32*32)
            
            two_hot_labels[n,int(labels[n])] = 1.0-ratio
            # print("two_hot_labels1",two_hot_labels)
            two_hot_labels[n,int(labels[n+1])] += ratio
            # print("two_hot_labels2",two_hot_labels)
        else:
            two_hot_labels[n,int(labels[n])] = 1.0

    two_hot_labels[imgs.shape[0]-1,int(labels[imgs.shape[0]-1])] = 1.0
    return imgs, two_hot_labels

def CutMix(imgs, labels, p, Cfg):
        
        two_hot_labels = torch.zeros((imgs.shape[0],Cfg.num_class)).to(Cfg.device)

        for n in range(imgs.shape[0]-1):
            if torch.rand(1) < p:

                original_image = imgs[n,:,:,:].to(Cfg.device)
                patch_image = imgs[n+1,:,:,:].to(Cfg.device)


                beta = 1.0                
                r = np.random.rand(1)
                mask = torch.ones((32,32)).to(Cfg.device)

                def rand_bbox(lam):
                    W = 32
                    H = 32
                    cut_rat = np.sqrt(1. - lam)
                    cut_w = np.int(W * cut_rat)
                    cut_h = np.int(H * cut_rat)

                    # uniform
                    cx = np.random.randint(W)
                    cy = np.random.randint(H)

                    bbx1 = np.clip(cx - cut_w // 2, 0, W)
                    bby1 = np.clip(cy - cut_h // 2, 0, H)
                    bbx2 = np.clip(cx + cut_w // 2, 0, W)
                    bby2 = np.clip(cy + cut_h // 2, 0, H)

                    return bbx1, bby1, bbx2, bby2
                
                bbx1, bby1, bbx2, bby2 = rand_bbox(np.random.beta(beta, beta))
                mask[bbx1:bbx2, bby1:bby2] = 0

                patch_mask = mask.clone().detach()
                patch_mask_inv = 1 - patch_mask.clone().detach()

                new_image = torch.zeros_like(imgs[n,:,:,:]).to(Cfg.device)

                # patch_mask = TF.rotate(patch_mask, int(angle))

                new_image[:] += patch_image[:] * patch_mask
                new_image[:] += original_image[:] * patch_mask_inv
                imgs[n] = new_image

                
                ratio = (torch.sum(patch_mask))/(32*32)
                ratio2 = (torch.sum(patch_mask_inv))/(32*32)
                
                two_hot_labels[n,int(labels[n])] = 1.0-ratio
                # print("two_hot_labels1",two_hot_labels)
                two_hot_labels[n,int(labels[n+1])] += ratio
                # print("two_hot_labels2",two_hot_labels)
            else:
                two_hot_labels[n,int(labels[n])] = 1.0

        two_hot_labels[imgs.shape[0]-1,int(labels[imgs.shape[0]-1])] = 1.0
        return imgs, two_hot_labels

def SmoothMix(imgs, labels, p, Cfg):

    two_hot_labels = torch.zeros((imgs.shape[0],Cfg.num_class)).to(Cfg.device)
    
    for n in range(imgs.shape[0]-1):
        if torch.rand(1) < p:

            original_image = imgs[n,:,:,:].to(Cfg.device)
            patch_image = imgs[n+1,:,:,:].to(Cfg.device)

            def gkern(kernlen, nsig):
                x = np.linspace(-nsig, nsig, kernlen+1)
                kern1d = np.diff(st.norm.cdf(x))
                kern2d = np.outer(kern1d, kern1d)
                return kern2d/kern2d.sum()

            dis = float(np.random.randint(20,40, size=1))
            mag = float(np.random.randint(750,1500, size=1))
            shift_x, shifr_y = np.random.randint(-4,5, size=2)
            kernel = gkern(24, dis/10)*mag
            kernel[kernel > 1.0] = 1.0
            kernel = np.pad(kernel,4)
            kernel = np.roll(kernel, shift_x, axis=1)
            kernel = np.roll(kernel, shifr_y, axis=0)
            patch_mask = torch.tensor(kernel).to(Cfg.device)
            patch_mask_inv = 1 - patch_mask.clone().detach().to(Cfg.device)

            new_image = torch.zeros_like(imgs[n,:,:,:]).to(Cfg.device)

            new_image[:] += patch_image[:] * patch_mask
            new_image[:] += original_image[:] * patch_mask_inv
            imgs[n] = new_image

            ratio = (torch.sum(patch_mask))/(32*32)
            
            two_hot_labels[n,int(labels[n])] = 1.0-ratio
            two_hot_labels[n,int(labels[n+1])] += ratio
        else:
            two_hot_labels[n,int(labels[n])] = 1.0

    two_hot_labels[imgs.shape[0]-1,int(labels[imgs.shape[0]-1])] = 1.0
    return imgs, two_hot_labels

###################################################################################
####   FMix

import torch.nn.functional as F
from fmix import sample_mask, FMixBase
import torch


def fmix_loss(input, y1, index, lam, train=True, reformulate=False):
    r"""Criterion for fmix

    Args:
        input: If train, mixed input. If not train, standard input
        y1: Targets for first image
        index: Permutation for mixing
        lam: Lambda value of mixing
        train: If true, sum cross entropy of input with y1 and y2, weighted by lam/(1-lam). If false, cross entropy loss with y1
    """

    if train and not reformulate:
        y2 = y1[index]
        return F.cross_entropy(input, y1) * lam + F.cross_entropy(input, y2) * (1 - lam)
    else:
        return F.cross_entropy(input, y1)


class FMix(FMixBase):
    """ FMix augmentation

        Args:
            decay_power (float): Decay power for frequency decay prop 1/f**d
            alpha (float): Alpha value for beta distribution from which to sample mean of mask
            size ([int] | [int, int] | [int, int, int]): Shape of desired mask, list up to 3 dims
            max_soft (float): Softening value between 0 and 0.5 which smooths hard edges in the mask.
            reformulate (bool): If True, uses the reformulation of [1].

        Example
        -------

        .. code-block:: python

            class FMixExp(pl.LightningModule):
                def __init__(*args, **kwargs):
                    self.fmix = Fmix(...)
                    # ...

                def training_step(self, batch, batch_idx):
                    x, y = batch
                    x = self.fmix(x)

                    feature_maps = self.forward(x)
                    logits = self.classifier(feature_maps)
                    loss = self.fmix.loss(logits, y)

                    # ...
                    return loss
    """
    def __init__(self, decay_power=3, alpha=1, size=(32, 32), max_soft=0.0, reformulate=False):
        super().__init__(decay_power, alpha, size, max_soft, reformulate)

    def __call__(self, x):
        # Sample mask and generate random permutation
        
        lam, mask = sample_mask(self.alpha, self.decay_power, self.size, self.max_soft, self.reformulate)
        index = torch.randperm(x.size(0)).to(x.device)
        mask = torch.from_numpy(mask).float().to(x.device)

        # Mix the images
        x1 = mask * x
        x2 = (1 - mask) * x[index]
        self.index = index
        self.lam = lam
        # print(lam)
        return x1+x2

    def loss(self, y_pred, y, train=True):
        return fmix_loss(y_pred, y, self.index, self.lam, train, self.reformulate)

#####################################################################################
###### Semantic CutMix

aam_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine((-45,45), translate=(0.3,0.3), scale=(0.7,1.3), shear=(0.8)),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
])
def AAM(imgs, labels, p, Cfg):
    imgs = imgs.to(Cfg.device)
    new_imgs = torch.zeros_like(imgs).to(Cfg.device)
    two_hot_labels = torch.zeros((imgs.shape[0],Cfg.num_class)).to(Cfg.device)
    pad = torch.nn.ZeroPad2d(Cfg.image_size)

    batch_size = imgs.shape[0]

    for n in range(batch_size):
        if torch.rand(1) < p:
            ratio,ratio2 = 0.0,0.0        
            while ratio < 0.4 or ratio2 < 0.4:
                dis = float(np.random.randint(5,100, size=1))#1,200
                mag = float(np.random.randint(50,3000, size=1))
                shift_x, shifr_y = np.random.randint(low = -(Cfg.image_size//2), high = (Cfg.image_size//2)+1, size=2)
                shift_x = shift_x * (1 if random.random() < 0.5 else -1)
                shifr_y = shifr_y * (1 if random.random() < 0.5 else -1)
                step = int(np.random.randint(Cfg.image_size//4,Cfg.image_size+1, size=1))

                kernel = torch.tensor(gkern(Cfg.image_size*2, dis/10,mag)).to(Cfg.device)
                #kernel = gkern_torch(64, dis/10, mag).to(device)

                kernel = torch.clamp(kernel, min = 0.0, max = 1.0)
                kernel = pad(kernel)

                kernel = torch.roll(kernel, shift_x, 1)

                kernel = torch.roll(kernel, shifr_y, 0)

                mask = torch.zeros((Cfg.image_size*2,Cfg.image_size*2),device = Cfg.device)

                for i in range(-Cfg.image_size,Cfg.image_size+1,step):
                    for j in range(-Cfg.image_size,Cfg.image_size+1,step):
                        mask += kernel[Cfg.image_size-i:Cfg.image_size*3-i,Cfg.image_size-j:Cfg.image_size*3-j]

                
                mask = torch.clamp(mask, min = 0.0, max = 1.0)
                
                mask = aam_tf(torch.unsqueeze(mask, 0))
                mask = mask[:,Cfg.image_size//2:int(Cfg.image_size*1.5),Cfg.image_size//2:int(Cfg.image_size*1.5)]

                patch_mask_inv = 1 - mask

                weight_mask = Cfg.cam_mask.to(Cfg.device)
                ratio = (torch.sum(mask*weight_mask))/(Cfg.image_size*Cfg.image_size)
                ratio2 = (torch.sum(patch_mask_inv*weight_mask))/(Cfg.image_size*Cfg.image_size)

            target_id = int(np.random.randint(0,imgs.shape[0]-1, size=1)) 
            two_hot_labels[n,int(labels[n])] = ratio
            two_hot_labels[n,int(labels[target_id])] += ratio2
            new_imgs[n] = (imgs[n,:,:,:] * mask + imgs[target_id,:,:,:] * patch_mask_inv)

            new_imgs[imgs.shape[0]-1] = imgs[imgs.shape[0]-1]
            two_hot_labels[imgs.shape[0]-1,int(labels[imgs.shape[0]-1])] = 1.0


        else:
            new_imgs = imgs
            two_hot_labels[n,int(labels[n])] = 1.0

    return new_imgs, two_hot_labels    

def AAM_withoutWCW(imgs, labels, p, Cfg):
    imgs = imgs.to(Cfg.device)
    new_imgs = torch.zeros_like(imgs).to(Cfg.device)
    two_hot_labels = torch.zeros((imgs.shape[0],Cfg.num_class)).to(Cfg.device)
    pad = torch.nn.ZeroPad2d(Cfg.image_size)

    batch_size = imgs.shape[0]

    for n in range(batch_size):
        if torch.rand(1) < p:
            ratio,ratio2 = 0.0,0.0
            #if True:
            # iii = 0
            # while ratio < 0.3 or ratio > 0.7 or ratio2 < 0.3 or ratio2 > 0.7:
            while ratio < 0.3 or ratio2 < 0.3:
                dis = float(np.random.randint(5,100, size=1))#1,200
                mag = float(np.random.randint(500,3000, size=1))
                shift_x, shifr_y = np.random.randint(low = -(Cfg.image_size//2), high = (Cfg.image_size//2)+1, size=2)
                shift_x = shift_x * (1 if random.random() < 0.5 else -1)
                shifr_y = shifr_y * (1 if random.random() < 0.5 else -1)
                step = int(np.random.randint(Cfg.image_size//2,Cfg.image_size+1, size=1))

                kernel = torch.tensor(gkern(Cfg.image_size*2, dis/10,mag)).to(Cfg.device)
                #kernel = gkern_torch(64, dis/10, mag).to(device)

                kernel = torch.clamp(kernel, min = 0.0, max = 1.0)
                kernel = pad(kernel)

                kernel = torch.roll(kernel, shift_x, 1)

                kernel = torch.roll(kernel, shifr_y, 0)

                mask = torch.zeros((Cfg.image_size*2,Cfg.image_size*2),device = Cfg.device)

                for i in range(-Cfg.image_size,Cfg.image_size+1,step):
                    for j in range(-Cfg.image_size,Cfg.image_size+1,step):
                        mask += kernel[Cfg.image_size-i:Cfg.image_size*3-i,Cfg.image_size-j:Cfg.image_size*3-j]

                
                mask = torch.clamp(mask, min = 0.0, max = 1.0)
                
                mask = aam_tf(torch.unsqueeze(mask, 0))
                mask = mask[:,Cfg.image_size//2:int(Cfg.image_size*1.5),Cfg.image_size//2:int(Cfg.image_size*1.5)]

                patch_mask_inv = 1 - mask

                target_id = int(np.random.randint(0,imgs.shape[0]-1, size=1))

                ratio = (torch.sum(mask))/(Cfg.image_size*Cfg.image_size)
                ratio2 = (torch.sum(patch_mask_inv))/(Cfg.image_size*Cfg.image_size)
                # print(iii)
                # iii+=1

                
            two_hot_labels[n,int(labels[n])] = ratio
            two_hot_labels[n,int(labels[target_id])] += ratio2
            new_imgs[n] = (imgs[n,:,:,:] * mask + imgs[target_id,:,:,:] * patch_mask_inv)

            new_imgs[imgs.shape[0]-1] = imgs[imgs.shape[0]-1]
            two_hot_labels[imgs.shape[0]-1,int(labels[imgs.shape[0]-1])] = 1.0


        else:
            new_imgs = imgs
            two_hot_labels[n,int(labels[n])] = 1.0

    return new_imgs, two_hot_labels    


 

def AAM_batch(imgs, labels, p, Cfg):
    imgs = imgs.to(Cfg.device)
    new_imgs = torch.zeros_like(imgs).to(Cfg.device)
    two_hot_labels = torch.zeros((imgs.shape[0],Cfg.num_class)).to(Cfg.device)
    pad = torch.nn.ZeroPad2d(Cfg.image_size)

    batch_size = imgs.shape[0]

    if torch.rand(1) < p:
        ratio,ratio2 = 0.0,0.0
        while ratio < 0.3 or ratio2 < 0.3:
            dis = float(np.random.randint(5,100, size=1))#1,200
            mag = float(np.random.randint(500,3000, size=1))
            shift_x, shifr_y = np.random.randint(low = -(Cfg.image_size//2), high = (Cfg.image_size//2)+1, size=2)
            shift_x = shift_x * (1 if random.random() < 0.5 else -1)
            shifr_y = shifr_y * (1 if random.random() < 0.5 else -1)
            step = int(np.random.randint(Cfg.image_size//2,Cfg.image_size+1, size=1))

            kernel = torch.tensor(gkern(Cfg.image_size*2, dis/10,mag)).to(Cfg.device)
            #kernel = gkern_torch(64, dis/10, mag).to(device)

            kernel = torch.clamp(kernel, min = 0.0, max = 1.0)
            kernel = pad(kernel)

            kernel = torch.roll(kernel, shift_x, 1)

            kernel = torch.roll(kernel, shifr_y, 0)

            mask = torch.zeros((Cfg.image_size*2,Cfg.image_size*2),device = Cfg.device)

            for i in range(-Cfg.image_size,Cfg.image_size+1,step):
                for j in range(-Cfg.image_size,Cfg.image_size+1,step):
                    mask += kernel[Cfg.image_size-i:Cfg.image_size*3-i,Cfg.image_size-j:Cfg.image_size*3-j]

            
            mask = torch.clamp(mask, min = 0.0, max = 1.0)
            
            mask = aam_tf(torch.unsqueeze(mask, 0))
            mask = mask[:,Cfg.image_size//2:int(Cfg.image_size*1.5),Cfg.image_size//2:int(Cfg.image_size*1.5)]

            patch_mask_inv = 1 - mask

            weight_mask = torch.tensor(gkern(Cfg.image_size*2,3,1350)).to(Cfg.device)
            weight_mask = weight_mask[Cfg.image_size//2:int(Cfg.image_size*1.5),Cfg.image_size//2:int(Cfg.image_size*1.5)]
            ratio = (torch.sum(mask*weight_mask))/(Cfg.image_size*Cfg.image_size)
            ratio2 = (torch.sum(patch_mask_inv*weight_mask))/(Cfg.image_size*Cfg.image_size)

        for n in range(imgs.shape[0]-1):
            target_id = int(np.random.randint(0,imgs.shape[0]-1, size=1))
            two_hot_labels[n,int(labels[n])] = ratio
            two_hot_labels[n,int(labels[target_id])] += ratio2
            new_imgs[n] = (imgs[n,:,:,:] * mask + imgs[target_id,:,:,:] * patch_mask_inv)

        new_imgs[imgs.shape[0]-1] = imgs[imgs.shape[0]-1]
        two_hot_labels[imgs.shape[0]-1,int(labels[imgs.shape[0]-1])] = 1.0

    else:
        new_imgs = imgs
        for n in range(imgs.shape[0]):
            two_hot_labels[n,int(labels[n])] = 1.0
        

    return new_imgs, two_hot_labels    

def AAM_batch_cam_weight(imgs, labels, p, Cfg):
    imgs = imgs.to(Cfg.device)
    new_imgs = torch.zeros_like(imgs).to(Cfg.device)
    two_hot_labels = torch.zeros((imgs.shape[0],Cfg.num_class)).to(Cfg.device)
    pad = torch.nn.ZeroPad2d(Cfg.image_size)

    batch_size = imgs.shape[0]

    if torch.rand(1) < p:
        ratio,ratio2 = 0.0,0.0
        while ratio < 0.3 or ratio2 < 0.3:
            dis = float(np.random.randint(5,100, size=1))#1,200
            mag = float(np.random.randint(500,3000, size=1))
            shift_x, shifr_y = np.random.randint(low = -(Cfg.image_size//2), high = (Cfg.image_size//2)+1, size=2)
            shift_x = shift_x * (1 if random.random() < 0.5 else -1)
            shifr_y = shifr_y * (1 if random.random() < 0.5 else -1)
            step = int(np.random.randint(Cfg.image_size//2,Cfg.image_size+1, size=1))

            kernel = torch.tensor(gkern(Cfg.image_size*2, dis/10,mag)).to(Cfg.device)
            #kernel = gkern_torch(64, dis/10, mag).to(device)

            kernel = torch.clamp(kernel, min = 0.0, max = 1.0)
            kernel = pad(kernel)

            kernel = torch.roll(kernel, shift_x, 1)

            kernel = torch.roll(kernel, shifr_y, 0)

            mask = torch.zeros((Cfg.image_size*2,Cfg.image_size*2),device = Cfg.device)

            for i in range(-Cfg.image_size,Cfg.image_size+1,step):
                for j in range(-Cfg.image_size,Cfg.image_size+1,step):
                    mask += kernel[Cfg.image_size-i:Cfg.image_size*3-i,Cfg.image_size-j:Cfg.image_size*3-j]

            
            mask = torch.clamp(mask, min = 0.0, max = 1.0)
            
            mask = aam_tf(torch.unsqueeze(mask, 0))
            mask = mask[:,Cfg.image_size//2:int(Cfg.image_size*1.5),Cfg.image_size//2:int(Cfg.image_size*1.5)]

            patch_mask_inv = 1 - mask

            weight_mask = Cfg.cam_mask.to(Cfg.device)
            ratio = (torch.sum(mask*weight_mask))/(Cfg.image_size*Cfg.image_size)
            ratio2 = (torch.sum(patch_mask_inv*weight_mask))/(Cfg.image_size*Cfg.image_size)

        for n in range(imgs.shape[0]-1):
            target_id = int(np.random.randint(0,imgs.shape[0]-1, size=1))
            two_hot_labels[n,int(labels[n])] = ratio
            two_hot_labels[n,int(labels[target_id])] += ratio2
            new_imgs[n] = (imgs[n,:,:,:] * mask + imgs[target_id,:,:,:] * patch_mask_inv)

        new_imgs[imgs.shape[0]-1] = imgs[imgs.shape[0]-1]
        two_hot_labels[imgs.shape[0]-1,int(labels[imgs.shape[0]-1])] = 1.0

    else:
        new_imgs = imgs
        for n in range(imgs.shape[0]):
            two_hot_labels[n,int(labels[n])] = 1.0
        

    return new_imgs, two_hot_labels    

def AAM_batch_ft(imgs, labels, p, Cfg):
    imgs = imgs.to(Cfg.device)
    new_imgs = torch.zeros_like(imgs).to(Cfg.device)
    two_hot_labels = torch.zeros((imgs.shape[0],Cfg.num_class)).to(Cfg.device)
    pad = torch.nn.ZeroPad2d(Cfg.image_size)

    batch_size = imgs.shape[0]

    if torch.rand(1) < p:
        ratio,ratio2 = 0.0,0.0
        while ratio < 0.4 or ratio2 < 0.4:
            dis = float(np.random.randint(5,100, size=1))#1,200
            mag = float(np.random.randint(50,3000, size=1))
            shift_x, shifr_y = np.random.randint(low = -(Cfg.image_size//2), high = (Cfg.image_size//2)+1, size=2)
            shift_x = shift_x * (1 if random.random() < 0.5 else -1)
            shifr_y = shifr_y * (1 if random.random() < 0.5 else -1)
            step = int(np.random.randint(Cfg.image_size//4,Cfg.image_size+1, size=1))

            kernel = torch.tensor(gkern(Cfg.image_size*2, dis/10,mag)).to(Cfg.device)
            #kernel = gkern_torch(64, dis/10, mag).to(device)

            kernel = torch.clamp(kernel, min = 0.0, max = 1.0)
            kernel = pad(kernel)

            kernel = torch.roll(kernel, shift_x, 1)

            kernel = torch.roll(kernel, shifr_y, 0)

            mask = torch.zeros((Cfg.image_size*2,Cfg.image_size*2),device = Cfg.device)

            for i in range(-Cfg.image_size,Cfg.image_size+1,step):
                for j in range(-Cfg.image_size,Cfg.image_size+1,step):
                    mask += kernel[Cfg.image_size-i:Cfg.image_size*3-i,Cfg.image_size-j:Cfg.image_size*3-j]

            
            mask = torch.clamp(mask, min = 0.0, max = 1.0)
            
            mask = aam_tf(torch.unsqueeze(mask, 0))
            mask = mask[:,Cfg.image_size//2:int(Cfg.image_size*1.5),Cfg.image_size//2:int(Cfg.image_size*1.5)]

            patch_mask_inv = 1 - mask

            weight_mask = torch.tensor(gkern(Cfg.image_size*2,3,1350)).to(Cfg.device)
            weight_mask = weight_mask[Cfg.image_size//2:int(Cfg.image_size*1.5),Cfg.image_size//2:int(Cfg.image_size*1.5)]
            ratio = (torch.sum(mask*weight_mask))/(Cfg.image_size*Cfg.image_size)
            ratio2 = (torch.sum(patch_mask_inv*weight_mask))/(Cfg.image_size*Cfg.image_size)

        for n in range(imgs.shape[0]-1):
            target_id = int(np.random.randint(0,imgs.shape[0]-1, size=1))
            two_hot_labels[n,int(labels[n])] = ratio
            two_hot_labels[n,int(labels[target_id])] += ratio2
            new_imgs[n] = (imgs[n,:,:,:] * mask + imgs[target_id,:,:,:] * patch_mask_inv)

        new_imgs[imgs.shape[0]-1] = imgs[imgs.shape[0]-1]
        two_hot_labels[imgs.shape[0]-1,int(labels[imgs.shape[0]-1])] = 1.0

    else:
        new_imgs = imgs
        for n in range(imgs.shape[0]):
            two_hot_labels[n,int(labels[n])] = 1.0
        

    return new_imgs, two_hot_labels   


def AugmentedSmoothMask(imgs, labels, p, Cfg):
    imgs = imgs.to(Cfg.device)
    new_imgs = torch.zeros_like(imgs).to(Cfg.device)
    two_hot_labels = torch.zeros((imgs.shape[0],Cfg.num_class)).to(Cfg.device)
    pad = torch.nn.ZeroPad2d(Cfg.image_size)

    batch_size = imgs.shape[0]

    if torch.rand(1) < p:
        ratio,ratio2 = 0.0,0.0
        while ratio < 0.4 or ratio2 < 0.4:
            dis = float(np.random.randint(5,100, size=1))#1,200
            mag = float(np.random.randint(50,3000, size=1))
            shift_x, shifr_y = np.random.randint(low = -(Cfg.image_size//2), high = (Cfg.image_size//2)+1, size=2)
            shift_x = shift_x * (1 if random.random() < 0.5 else -1)
            shifr_y = shifr_y * (1 if random.random() < 0.5 else -1)
            step = int(np.random.randint(Cfg.image_size//4,Cfg.image_size+1, size=1))

            kernel = torch.tensor(gkern(Cfg.image_size*2, dis/10,mag)).to(Cfg.device)
            #kernel = gkern_torch(64, dis/10, mag).to(device)

            kernel = torch.clamp(kernel, min = 0.0, max = 1.0)
            kernel = pad(kernel)

            kernel = torch.roll(kernel, shift_x, 1)

            kernel = torch.roll(kernel, shifr_y, 0)

            mask = torch.zeros((Cfg.image_size*2,Cfg.image_size*2),device = Cfg.device)

            for i in range(-Cfg.image_size,Cfg.image_size+1,step):
                for j in range(-Cfg.image_size,Cfg.image_size+1,step):
                    mask += kernel[Cfg.image_size-i:Cfg.image_size*3-i,Cfg.image_size-j:Cfg.image_size*3-j]

            
            mask = torch.clamp(mask, min = 0.0, max = 1.0)
            
            mask = aam_tf(torch.unsqueeze(mask, 0))
            mask = mask[:,Cfg.image_size//2:int(Cfg.image_size*1.5),Cfg.image_size//2:int(Cfg.image_size*1.5)]

            patch_mask_inv = 1 - mask

            ratio = (torch.sum(mask))/(Cfg.image_size*Cfg.image_size)
            ratio2 = (torch.sum(patch_mask_inv))/(Cfg.image_size*Cfg.image_size)

        for n in range(imgs.shape[0]-1):
            target_id = int(np.random.randint(0,imgs.shape[0]-1, size=1))
            two_hot_labels[n,int(labels[n])] = ratio
            two_hot_labels[n,int(labels[target_id])] += ratio2
            new_imgs[n] = (imgs[n,:,:,:] * mask + imgs[target_id,:,:,:] * patch_mask_inv)

        new_imgs[imgs.shape[0]-1] = imgs[imgs.shape[0]-1]
        two_hot_labels[imgs.shape[0]-1,int(labels[imgs.shape[0]-1])] = 1.0

    else:
        new_imgs = imgs
        for n in range(imgs.shape[0]):
            two_hot_labels[n,int(labels[n])] = 1.0
        

    return new_imgs, two_hot_labels   


aam_tf_ft = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine((-45,45), translate=(0.3,0.3), scale=(0.9,1.1), shear=(0.9)),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
])
def AAM_batch_ft_camWeight(imgs, labels, p, Cfg):
    imgs = imgs.to(Cfg.device)
    new_imgs = torch.zeros_like(imgs).to(Cfg.device)
    two_hot_labels = torch.zeros((imgs.shape[0],Cfg.num_class)).to(Cfg.device)
    pad = torch.nn.ZeroPad2d(Cfg.image_size)

    batch_size = imgs.shape[0]

    if torch.rand(1) < p:
        ratio,ratio2 = 0.0,0.0
        while ratio < 0.4 or ratio2 < 0.4:
            dis = float(np.random.randint(5,100, size=1))#1,200
            mag = float(np.random.randint(50,3000, size=1))
            shift_x, shifr_y = np.random.randint(low = -(Cfg.image_size//2), high = (Cfg.image_size//2)+1, size=2)
            shift_x = shift_x * (1 if random.random() < 0.5 else -1)
            shifr_y = shifr_y * (1 if random.random() < 0.5 else -1)
            step = int(np.random.randint(Cfg.image_size//4,Cfg.image_size+1, size=1))

            kernel = torch.tensor(gkern(Cfg.image_size*2, dis/10,mag)).to(Cfg.device)
            #kernel = gkern_torch(64, dis/10, mag).to(device)

            kernel = torch.clamp(kernel, min = 0.0, max = 1.0)
            kernel = pad(kernel)

            kernel = torch.roll(kernel, shift_x, 1)

            kernel = torch.roll(kernel, shifr_y, 0)

            mask = torch.zeros((Cfg.image_size*2,Cfg.image_size*2),device = Cfg.device)

            for i in range(-Cfg.image_size,Cfg.image_size+1,step):
                for j in range(-Cfg.image_size,Cfg.image_size+1,step):
                    mask += kernel[Cfg.image_size-i:Cfg.image_size*3-i,Cfg.image_size-j:Cfg.image_size*3-j]

            
            mask = torch.clamp(mask, min = 0.0, max = 1.0)
            
            mask = aam_tf(torch.unsqueeze(mask, 0))
            mask = mask[:,Cfg.image_size//2:int(Cfg.image_size*1.5),Cfg.image_size//2:int(Cfg.image_size*1.5)]

            patch_mask_inv = 1 - mask

            weight_mask = Cfg.cam_mask.to(Cfg.device)
            ratio = (torch.sum(mask*weight_mask))/(Cfg.image_size*Cfg.image_size)
            ratio2 = (torch.sum(patch_mask_inv*weight_mask))/(Cfg.image_size*Cfg.image_size)

        for n in range(imgs.shape[0]-1):
            target_id = int(np.random.randint(0,imgs.shape[0]-1, size=1))
            two_hot_labels[n,int(labels[n])] = ratio
            two_hot_labels[n,int(labels[target_id])] += ratio2
            new_imgs[n] = (imgs[n,:,:,:] * mask + imgs[target_id,:,:,:] * patch_mask_inv)

        new_imgs[imgs.shape[0]-1] = imgs[imgs.shape[0]-1]
        two_hot_labels[imgs.shape[0]-1,int(labels[imgs.shape[0]-1])] = 1.0

    else:
        new_imgs = imgs
        for n in range(imgs.shape[0]):
            two_hot_labels[n,int(labels[n])] = 1.0
        

    return new_imgs, two_hot_labels   
aam_tf_ft = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine((-45,45), translate=(0.3,0.3), scale=(0.9,1.1), shear=(0.9)),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
])
def AAM_batch_ftft(imgs, labels, p, Cfg):
    imgs = imgs.to(Cfg.device)
    new_imgs = torch.zeros_like(imgs).to(Cfg.device)
    two_hot_labels = torch.zeros((imgs.shape[0],Cfg.num_class)).to(Cfg.device)
    pad = torch.nn.ZeroPad2d(Cfg.image_size)

    batch_size = imgs.shape[0]

    if torch.rand(1) < p:
        ratio,ratio2 = 0.0,0.0
        while ratio < 0.4 or ratio2 < 0.4:
            dis = float(np.random.randint(5,100, size=1))#1,200
            mag = float(np.random.randint(50,3000, size=1))
            shift_x, shifr_y = np.random.randint(low = -(Cfg.image_size//2), high = (Cfg.image_size//2)+1, size=2)
            shift_x = shift_x * (1 if random.random() < 0.5 else -1)
            shifr_y = shifr_y * (1 if random.random() < 0.5 else -1)
            step = int(np.random.randint(Cfg.image_size//2,Cfg.image_size*2+1, size=1))

            kernel = torch.tensor(gkern(Cfg.image_size*2, dis/10,mag)).to(Cfg.device)
            #kernel = gkern_torch(64, dis/10, mag).to(device)

            kernel = torch.clamp(kernel, min = 0.0, max = 1.0)
            kernel = pad(kernel)

            kernel = torch.roll(kernel, shift_x, 1)

            kernel = torch.roll(kernel, shifr_y, 0)

            mask = torch.zeros((Cfg.image_size*2,Cfg.image_size*2),device = Cfg.device)

            for i in range(-Cfg.image_size,Cfg.image_size+1,step):
                for j in range(-Cfg.image_size,Cfg.image_size+1,step):
                    mask += kernel[Cfg.image_size-i:Cfg.image_size*3-i,Cfg.image_size-j:Cfg.image_size*3-j]

            
            mask = torch.clamp(mask, min = 0.0, max = 1.0)
            
            mask = aam_tf_ft(torch.unsqueeze(mask, 0))
            mask = mask[:,Cfg.image_size//2:int(Cfg.image_size*1.5),Cfg.image_size//2:int(Cfg.image_size*1.5)]

            patch_mask_inv = 1 - mask

            weight_mask = torch.tensor(gkern(Cfg.image_size*2,3,1350)).to(Cfg.device)
            weight_mask = weight_mask[Cfg.image_size//2:int(Cfg.image_size*1.5),Cfg.image_size//2:int(Cfg.image_size*1.5)]
            ratio = (torch.sum(mask*weight_mask))/(Cfg.image_size*Cfg.image_size)
            ratio2 = (torch.sum(patch_mask_inv*weight_mask))/(Cfg.image_size*Cfg.image_size)

        for n in range(imgs.shape[0]-1):
            target_id = int(np.random.randint(0,imgs.shape[0]-1, size=1))
            two_hot_labels[n,int(labels[n])] = ratio
            two_hot_labels[n,int(labels[target_id])] += ratio2
            new_imgs[n] = (imgs[n,:,:,:] * mask + imgs[target_id,:,:,:] * patch_mask_inv)

        new_imgs[imgs.shape[0]-1] = imgs[imgs.shape[0]-1]
        two_hot_labels[imgs.shape[0]-1,int(labels[imgs.shape[0]-1])] = 1.0

    else:
        new_imgs = imgs
        for n in range(imgs.shape[0]):
            two_hot_labels[n,int(labels[n])] = 1.0
        

    return new_imgs, two_hot_labels    

def AAM_batch_withoutWCW(imgs, labels, p, Cfg):
    imgs = imgs.to(Cfg.device)
    new_imgs = torch.zeros_like(imgs).to(Cfg.device)
    two_hot_labels = torch.zeros((imgs.shape[0],Cfg.num_class)).to(Cfg.device)
    pad = torch.nn.ZeroPad2d(Cfg.image_size)

    batch_size = imgs.shape[0]

    if torch.rand(1) < p:
        ratio,ratio2 = 0.0,0.0
        while ratio < 0.3 or ratio2 < 0.3:
            dis = float(np.random.randint(5,100, size=1))#1,200
            mag = float(np.random.randint(500,3000, size=1))
            shift_x, shifr_y = np.random.randint(low = -(Cfg.image_size//2), high = (Cfg.image_size//2)+1, size=2)
            shift_x = shift_x * (1 if random.random() < 0.5 else -1)
            shifr_y = shifr_y * (1 if random.random() < 0.5 else -1)
            step = int(np.random.randint(Cfg.image_size//2,Cfg.image_size+1, size=1))

            kernel = torch.tensor(gkern(Cfg.image_size*2, dis/10,mag)).to(Cfg.device)
            #kernel = gkern_torch(64, dis/10, mag).to(device)

            kernel = torch.clamp(kernel, min = 0.0, max = 1.0)
            kernel = pad(kernel)

            kernel = torch.roll(kernel, shift_x, 1)

            kernel = torch.roll(kernel, shifr_y, 0)

            mask = torch.zeros((Cfg.image_size*2,Cfg.image_size*2),device = Cfg.device)

            for i in range(-Cfg.image_size,Cfg.image_size+1,step):
                for j in range(-Cfg.image_size,Cfg.image_size+1,step):
                    mask += kernel[Cfg.image_size-i:Cfg.image_size*3-i,Cfg.image_size-j:Cfg.image_size*3-j]

            
            mask = torch.clamp(mask, min = 0.0, max = 1.0)
            
            mask = aam_tf(torch.unsqueeze(mask, 0))
            mask = mask[:,Cfg.image_size//2:int(Cfg.image_size*1.5),Cfg.image_size//2:int(Cfg.image_size*1.5)]

            patch_mask_inv = 1 - mask

            ratio = (torch.sum(mask))/(Cfg.image_size*Cfg.image_size)
            ratio2 = (torch.sum(patch_mask_inv))/(Cfg.image_size*Cfg.image_size)

        for n in range(imgs.shape[0]-1):
            target_id = int(np.random.randint(0,imgs.shape[0]-1, size=1))
            two_hot_labels[n,int(labels[n])] = ratio
            two_hot_labels[n,int(labels[target_id])] += ratio2
            new_imgs[n] = (imgs[n,:,:,:] * mask + imgs[target_id,:,:,:] * patch_mask_inv)

        new_imgs[imgs.shape[0]-1] = imgs[imgs.shape[0]-1]
        two_hot_labels[imgs.shape[0]-1,int(labels[imgs.shape[0]-1])] = 1.0

    else:
        new_imgs = imgs
        for n in range(imgs.shape[0]):
            two_hot_labels[n,int(labels[n])] = 1.0
        

    return new_imgs, two_hot_labels    
    

def AAM_CAM(imgs, labels, p, Cfg):

    imgs = imgs.to(Cfg.device)
    new_imgs = torch.zeros_like(imgs).to(Cfg.device)
    two_hot_labels = torch.zeros((imgs.shape[0],Cfg.num_class)).to(Cfg.device)
    pad = torch.nn.ZeroPad2d(Cfg.image_size)

    batch_size = imgs.shape[0]

    for n in range(batch_size):
        if torch.rand(1) < p:
            ratio,ratio2 = 0.0,0.0
            while ratio < 0.3 or ratio2 < 0.3 or ratio + ratio2 < 0.6 or ratio + ratio > 1.4:
                dis = float(np.random.randint(5,100, size=1))#1,200
                mag = float(np.random.randint(50,3000, size=1))
                shift_x, shifr_y = np.random.randint(low = -(Cfg.image_size//2), high = (Cfg.image_size//2)+1, size=2)
                shift_x = shift_x * (1 if random.random() < 0.5 else -1)
                shifr_y = shifr_y * (1 if random.random() < 0.5 else -1)
                step = int(np.random.randint(Cfg.image_size//4,Cfg.image_size+1, size=1))

                kernel = torch.tensor(gkern(Cfg.image_size*2, dis/10,mag)).to(Cfg.device)
                #kernel = gkern_torch(64, dis/10, mag).to(device)

                kernel = torch.clamp(kernel, min = 0.0, max = 1.0)
                kernel = pad(kernel)

                kernel = torch.roll(kernel, shift_x, 1)

                kernel = torch.roll(kernel, shifr_y, 0)

                mask = torch.zeros((Cfg.image_size*2,Cfg.image_size*2),device = Cfg.device)

                for i in range(-Cfg.image_size,Cfg.image_size+1,step):
                    for j in range(-Cfg.image_size,Cfg.image_size+1,step):
                        mask += kernel[Cfg.image_size-i:Cfg.image_size*3-i,Cfg.image_size-j:Cfg.image_size*3-j]

                
                mask = torch.clamp(mask, min = 0.0, max = 1.0)
                
                mask = aam_tf(torch.unsqueeze(mask, 0))
                mask = mask[:,Cfg.image_size//2:int(Cfg.image_size*1.5),Cfg.image_size//2:int(Cfg.image_size*1.5)]

                patch_mask_inv = 1 - mask
                target_id = int(np.random.randint(0,imgs.shape[0], size=1))
                weight_mask1 = get_CAM(torch.unsqueeze(imgs[n,:,:,:],0), Cfg.cam_model, Cfg.image_size).to(Cfg.device)
                weight_mask2 = get_CAM(torch.unsqueeze(imgs[target_id,:,:,:],0), Cfg.cam_model, Cfg.image_size).to(Cfg.device)

                ratio = (torch.sum(mask*weight_mask1))/(Cfg.image_size*Cfg.image_size)
                ratio2 = (torch.sum(patch_mask_inv*weight_mask2))/(Cfg.image_size*Cfg.image_size)

                if n == 0:
                    plt.figure()
                    plt.imshow(weight_mask1.cpu())
                    plt.colorbar()
                    plt.savefig('./wm1.png')
                    print('ratio', ratio)

                    plt.figure()
                    plt.imshow(weight_mask2.cpu())
                    plt.colorbar()
                    plt.savefig('./wm2.png')
                    print('ratio', ratio2)

                    plt.figure()
                    plt.imshow(torch.squeeze(mask).cpu())
                    plt.colorbar()
                    plt.savefig('./m.png')
                

                
            two_hot_labels[n,int(labels[n])] = ratio
            two_hot_labels[n,int(labels[target_id])] += ratio2
            new_imgs[n] = (imgs[n,:,:,:] * mask + imgs[target_id,:,:,:] * patch_mask_inv)


        else:
            new_imgs = imgs
            two_hot_labels[n,int(labels[n])] = 1.0

    return new_imgs, two_hot_labels    

def get_image(args):
    index, imgs, labels, Cfg, p = args
    pad_fc = torch.nn.ZeroPad2d(Cfg.image_size)
    with torch.no_grad():
        if torch.rand(1) < p:
            ratio,ratio2 = 0.0,0.0
            while ratio < 0.3 or ratio2 < 0.3 or ratio + ratio2 < 0.6 or ratio + ratio > 1.4:
                dis = float(np.random.randint(5,100, size=1))#1,200
                mag = float(np.random.randint(50,3000, size=1))
                shift_x, shifr_y = np.random.randint(low = -(Cfg.image_size//2), high = (Cfg.image_size//2)+1, size=2)
                shift_x = shift_x * (1 if random.random() < 0.5 else -1)
                shifr_y = shifr_y * (1 if random.random() < 0.5 else -1)
                step = int(np.random.randint(Cfg.image_size//4,Cfg.image_size+1, size=1))

                kernel = torch.tensor(gkern(Cfg.image_size*2, dis/10,mag)).to(Cfg.device)
                #kernel = gkern_torch(64, dis/10, mag).to(device)

                kernel = torch.clamp(kernel, min = 0.0, max = 1.0)
                kernel = pad_fc(kernel)

                kernel = torch.roll(kernel, shift_x, 1)

                kernel = torch.roll(kernel, shifr_y, 0)

                mask = torch.zeros((Cfg.image_size*2,Cfg.image_size*2),device = Cfg.device)

                for i in range(-Cfg.image_size,Cfg.image_size+1,step):
                    for j in range(-Cfg.image_size,Cfg.image_size+1,step):
                        mask += kernel[Cfg.image_size-i:Cfg.image_size*3-i,Cfg.image_size-j:Cfg.image_size*3-j]

                
                mask = torch.clamp(mask, min = 0.0, max = 1.0)
                
                mask = aam_tf(torch.unsqueeze(mask, 0))
                mask = mask[:,Cfg.image_size//2:int(Cfg.image_size*1.5),Cfg.image_size//2:int(Cfg.image_size*1.5)]

                patch_mask_inv = 1 - mask
                target_id = int(np.random.randint(0,imgs.shape[0], size=1))
                weight_mask1 = get_CAM(torch.unsqueeze(imgs[index,:,:,:],0), Cfg.cam_model, Cfg.image_size).to(Cfg.device)
                weight_mask2 = get_CAM(torch.unsqueeze(imgs[target_id,:,:,:],0), Cfg.cam_model, Cfg.image_size).to(Cfg.device)
                ratio = (torch.sum(mask*weight_mask1))/(Cfg.image_size*Cfg.image_size)
                ratio2 = (torch.sum(patch_mask_inv*weight_mask2))/(Cfg.image_size*Cfg.image_size)

            two_hot_label = torch.zeros(Cfg.num_class).to(Cfg.device)
            two_hot_label[int(labels[index])] = ratio
            two_hot_label[int(labels[target_id])] += ratio2
            new_img = (imgs[index,:,:,:] * mask + imgs[target_id,:,:,:] * patch_mask_inv)
        else:
            two_hot_label = torch.zeros(Cfg.num_class).to(Cfg.device)
            new_img = imgs[index]
            two_hot_label[int(labels[index])] = 1.0
        new_img = torch.clone(new_img)
        two_hot_label = torch.clone(two_hot_label)
    return [new_img, two_hot_label]



def AAM_CAM_mp(imgs, labels, p, Cfg, pool):
    with torch.no_grad():
        imgs = imgs.to(Cfg.device)
        new_imgs = torch.zeros_like(imgs).to(Cfg.device)
        two_hot_labels = torch.zeros((imgs.shape[0],Cfg.num_class)).to(Cfg.device)

        batch_size = imgs.shape[0]
        # print(len([index for index in range(batch_size)]), \
        #       len([imgs for _ in range(batch_size)]), \
        #       len([labels for _ in range(batch_size)]), \
        #       len([Cfg for _ in range(batch_size)]), \
        #       len([p for _ in range(batch_size)]))
        args = zip([index for index in range(batch_size)], [imgs for _ in range(batch_size)], [labels for _ in range(batch_size)], [Cfg for _ in range(batch_size)], [p for _ in range(batch_size)])
        datas = pool.map(get_image, args)

        new_imgs = torch.stack([data[0] for data in datas]).to(Cfg.device)
        two_hot_labels = torch.stack([data[1] for data in datas]).to(Cfg.device)

        # print(new_imgs.shape, two_hot_labels.shape)
        return new_imgs, two_hot_labels  