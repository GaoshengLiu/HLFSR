import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
from skimage import metrics
import os
import numpy as np
import random
import math
import h5py
import copy
import imageio

class TrainSetLoader(Dataset):
    def __init__(self, cfg):
        super(TrainSetLoader, self).__init__()
        self.train_dir = cfg.trainset_dir
        self.seqfiles     = os.listdir(self.train_dir)
        self.seqlength    = len(self.seqfiles)
        self.scale = cfg.scale
        self.angRes = cfg.angRes
        self.patchsize = cfg.patchsize*cfg.scale
        scene_idx = []
        for i in range(150): ###300 for x4best, 400 for before models
            scene_idx = np.append(scene_idx, list(range(self.seqlength)))
        self.scene_idx = scene_idx.astype('int')
        self.item_num = len(self.scene_idx)

    def __getitem__(self, index):
        scene_id = self.scene_idx[index]
        #print(scene_id)
        scene_name = self.seqfiles[scene_id]
        scenefolder = os.path.join(self.train_dir, scene_name)
        #print(scenefolder)

        hf = h5py.File(scenefolder, 'r')
        label = hf.get('img_label') # [N,ah,aw,h,w]
        #self.img_LR_2 = hf.get('img_LR_2')   # [N,ah,aw,h/2,w/2]
        if self.scale==2:
            data = hf.get('img_LR_2')   # [N,ah,aw,h/4,w/4]
        elif self.scale==4:
            data = hf.get('img_LR_4')   # [N,ah,aw,h/4,w/4]
        elif self.scale==8:
            data = hf.get('img_LR_8')   # [N,ah,aw,h/8,w/8]
        else:
            print("No data prepared for scale:{}".format(self.scale))
        data, label = np.transpose(data, (1,0,3,2)), np.transpose(label, (1,0,3,2)) ## v u w h -> u,v,h,w

        u,v,h,w = label.shape
        an_crop = math.ceil((u-self.angRes)/2)
        data  = data[an_crop:an_crop+self.angRes, an_crop:an_crop+self.angRes, :, :]  # [ah,aw,ph,pw]
        label = label[an_crop:an_crop+self.angRes, an_crop:an_crop+self.angRes, :, :]  # [ah,aw,ph,pw]

        """ Data Augmentation """
        data, label = random_crop(data, label, self.patchsize, self.scale)
        data, label = augmentation(data, label)
        #data = data.reshape(-1, c, self.patchsize, self.patchsize)  # [an,3,ph,pw]
        center = self.angRes//2
        HR   = label[center,center,:,:]
        LR, HR, label = torch.from_numpy(data.astype(np.float32)/255.0), torch.from_numpy(HR.astype(np.float32)/255.0), torch.from_numpy(label.astype(np.float32)/255.0)
        return LR.unsqueeze(2), HR.unsqueeze(0), label.unsqueeze(2)

    def __len__(self):
        return self.item_num

def augmentation(x, y):
    if random.random() < 0.5:  # flip along U-H direction
        x = np.flip(np.flip(x, 0), 2)
        y = np.flip(np.flip(y, 0), 2)
    if random.random() < 0.5:  # flip along W-V direction
        x = np.flip(np.flip(x, 1), 3)
        y = np.flip(np.flip(y, 1), 3)
    if random.random() < 0.5: # transpose between U-V and H-W
        x = x.transpose(1, 0, 3, 2)
        y = y.transpose(1, 0, 3, 2)
    return x, y

def random_crop(lr, hr, psize, scale):
    angRes, angRes, h, w = hr.shape

    x = random.randrange(0, h - psize, 8)
    y = random.randrange(0, w - psize, 8)

    hr = hr[:, :, x:x+psize, y:y+psize] # [ah,aw,ph,pw]
    lr = lr[:, :, x//scale:x//scale+psize//scale, y//scale:y//scale+psize//scale] # [ah,aw,ph/2,pw/2] 

    return lr, hr

class ValSetLoader(Dataset):
    def __init__(self, cfg):
        super(ValSetLoader, self).__init__()
        self.validset_dir = cfg.validset_dir
        self.seqfiles     = os.listdir(self.validset_dir)
        self.seqlength    = len(self.seqfiles)       
        self.scale = cfg.scale
        self.angRes = cfg.angRes
        
    def __getitem__(self, index):
        seq_name = self.seqfiles[index]
        #print(seq_name)
        seqfolder = os.path.join(self.validset_dir, seq_name)

        hf = h5py.File(seqfolder, 'r')
        #self.GT_rgb = hf.get('/GT_rgb')  #[N,ah,aw,3,h,w]            
        gt_y = hf.get('/GT_y')
        lr = hf.get('/LR')
        lr, gt_y = np.transpose(lr, (1,0,3,2)), np.transpose(gt_y, (1,0,3,2)) ## v u w h -> u,v,h,w  

        u,v,h,w = gt_y.shape
        an_crop = math.ceil((u-self.angRes)/2)
       
        label = gt_y[an_crop:an_crop+self.angRes, an_crop:an_crop+self.angRes,:h,:w]#.reshape(-1,h,w)
        #lr_ycbcr_up = self.LR_ycbcr_up[index]
        #lr_ycbcr_up = lr_ycbcr_up[:self.an,:self.an,:,:h,:w].reshape(-1,3,h,w)
        lr = lr[an_crop:an_crop+self.angRes, an_crop:an_crop+self.angRes,:h//self.scale,:w//self.scale]
        #hr = self.HR[index]
        #hr = hr[0,0,:h,:w]#.reshape(-1,h,w)
        hr = label[self.angRes//2, self.angRes//2,:,:]#.reshape(-1,h,w)
         
        #gt_rgb = torch.from_numpy(gt_rgb.astype(np.float32)/255.0)
        label   = torch.from_numpy(label.astype(np.float32)/255.0)
        #lr_ycbcr_up = torch.from_numpy(lr_ycbcr_up.astype(np.float32)/255.0)
        lr = torch.from_numpy(lr.astype(np.float32)/255.0) 
        hr = torch.from_numpy(hr.astype(np.float32)/255.0)
        return lr.unsqueeze(2), hr.unsqueeze(0), label.unsqueeze(2)

    def __len__(self):
        return self.seqlength
    
class TestSetLoader(Dataset):
    def __init__(self, cfg):
        super(TestSetLoader, self).__init__()
        self.validset_dir = cfg.validset_dir
        self.seqfiles     = os.listdir(self.validset_dir)
        self.seqlength    = len(self.seqfiles)       
        self.scale = cfg.scale
        self.angRes = cfg.angRes

    def __getitem__(self, index):
        seq_name = self.seqfiles[index]
        #print(seq_name)
        seqfolder = os.path.join(self.validset_dir, seq_name)

        hf = h5py.File(seqfolder, 'r')
        #self.GT_rgb = hf.get('/GT_rgb')  #[N,ah,aw,3,h,w]            
        gt_y = hf.get('/GT_y')      #[N,aw,ah,h,w]
        LR_ycbcr_up = hf.get('/LR_up_ycbcr') #[N,ah,aw,3,h,w]
        lr = hf.get('/LR') #[N,ah,aw,h/s,w/s]
        lr, gt_y, LR_ycbcr_up = np.transpose(lr, (1,0,3,2)), np.transpose(gt_y, (1,0,3,2)), np.transpose(LR_ycbcr_up, (1,0,2,4,3)) ## v u w h -> u,v,h,w  

        u,v,h,w = gt_y.shape
        an_crop = math.ceil((u-self.angRes)/2)
       
        gt_y = gt_y[an_crop:an_crop+self.angRes, an_crop:an_crop+self.angRes,:h,:w]#.reshape(-1,h,w)
        #lr_ycbcr_up = self.LR_ycbcr_up[index]
        lr_ycbcr_up = LR_ycbcr_up[an_crop:an_crop+self.angRes,an_crop:an_crop+self.angRes,:,:h,:w]#.reshape(-1,3,h,w)
        lr = lr[an_crop:an_crop+self.angRes, an_crop:an_crop+self.angRes,:h//self.scale,:w//self.scale]
        #hr = self.HR[index]
        #hr = hr[0,0,:h,:w]#.reshape(-1,h,w)
        hr = gt_y[self.angRes//2,self.angRes//2,:,:]#.reshape(-1,h,w)
         
        #gt_rgb = torch.from_numpy(gt_rgb.astype(np.float32)/255.0)
        label   = torch.from_numpy(gt_y.astype(np.float32)/255.0)
        lr_ycbcr_up = torch.from_numpy(lr_ycbcr_up.astype(np.float32)/255.0)
        lr = torch.from_numpy(lr.astype(np.float32)/255.0) 
        hr = torch.from_numpy(hr.astype(np.float32)/255.0)
        return lr.unsqueeze(2), hr.unsqueeze(0), label.unsqueeze(2), lr_ycbcr_up

    def __len__(self):
        return self.seqlength
# def rgb2ycbcr(x):
#     y = np.zeros(x.shape, dtype='double')
#     y[:,:,0] =  65.481 * x[:, :, 0] + 128.553 * x[:, :, 1] +  24.966 * x[:, :, 2] +  16.0
#     y[:,:,1] = -37.797 * x[:, :, 0] -  74.203 * x[:, :, 1] + 112.000 * x[:, :, 2] + 128.0
#     y[:,:,2] = 112.000 * x[:, :, 0] -  93.786 * x[:, :, 1] -  18.214 * x[:, :, 2] + 128.0
#     y = y / 255.0
#     return y
# def ycbcr2rgb(x):
#     mat = np.array(
#         [[65.481, 128.553, 24.966],
#          [-37.797, -74.203, 112.0],
#          [112.0, -93.786, -18.214]])
#     mat_inv = np.linalg.inv(mat)
#     offset = np.matmul(mat_inv, np.array([16, 128, 128]))
#     mat_inv = mat_inv * 255
#     y = np.zeros(x.shape, dtype='double')
#     y[:,:,0] =  mat_inv[0,0] * x[:, :, 0] + mat_inv[0,1] * x[:, :, 1] + mat_inv[0,2] * x[:, :, 2] - offset[0]
#     y[:,:,1] =  mat_inv[1,0] * x[:, :, 0] + mat_inv[1,1] * x[:, :, 1] + mat_inv[1,2] * x[:, :, 2] - offset[1]
#     y[:,:,2] =  mat_inv[2,0] * x[:, :, 0] + mat_inv[2,1] * x[:, :, 1] + mat_inv[2,2] * x[:, :, 2] - offset[2]

#     return y
def ycbcr2rgb(ycbcr):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = copy.deepcopy(ycbcr)
    rgb[:,0] -= 16. / 255.
    rgb[:,1:] -= 128. / 255.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    return rgb.clip(0, 1).reshape(shape).astype(np.float32)
def cal_metrics(label, out):

    U, V, H, W = label.size()
    #label_y = (65.481 * label[:, :, 0, :, :] + 128.553 * label[:, :, 1, :, :] + 24.966 * label[:, :, 2, :, :] + 16) / 255.0
    #out_y = (65.481 * out[:, :, 0, :, :] + 128.553 * out[:, :, 1, :, :] + 24.966 * out[:, :, 2, :, :] + 16) / 255.0

    label = label.data.cpu().numpy().clip(0, 1)
    out   = out.data.cpu().numpy().clip(0, 1)

    PSNR = np.zeros(shape=(U, V), dtype='float32')
    SSIM = np.zeros(shape=(U, V), dtype='float32')
    center = U//2
    for u in range(U):
        for v in range(V):
            if u==center and v==center:
                continue
            else:                
                PSNR[u, v] = metrics.peak_signal_noise_ratio(label[u, v, :, :], out[u, v, :, :], data_range=1.0)
                SSIM[u, v] = metrics.structural_similarity(label[u, v, :, :], out[u, v, :, :], gaussian_weights=True, data_range=1.0)
    PSNR[center, center] = 0
    SSIM[center, center] = 0

    #print(np.sum(PSNR > 0))
    PSNR_mean = PSNR.sum() / np.sum(PSNR > 0)
    SSIM_mean = SSIM.sum() / np.sum(SSIM > 0)

    return PSNR_mean, SSIM_mean


def ImageExtend(Im, bdr):
    [_, _, h, w] = Im.size()
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[:, :, h - bdr[0]: 2 * h + bdr[1], w - bdr[2]: 2 * w + bdr[3]]

    return Im_out


def LFdivide(lf, patch_size, stride):
    U, V, C, H, W = lf.shape
    data = rearrange(lf, 'u v c h w -> (u v) c h w')

    bdr = (patch_size - stride) // 2
    numU = (H + bdr * 2 - 1) // stride
    numV = (W + bdr * 2 - 1) // stride
    data_pad = ImageExtend(data, [bdr, bdr + stride - 1, bdr, bdr + stride - 1])
    subLF = F.unfold(data_pad, kernel_size=patch_size, stride=stride)
    subLF = rearrange(subLF, '(u v) (c h w) (n1 n2) -> n1 n2 u v c h w',
                      n1=numU, n2=numV, u=U, v=V, h=patch_size, w=patch_size)

    return subLF


def LFintegrate(subLFs, patch_size, stride):
    n1, n2, u, v, c, h, w = subLFs.shape
    bdr = (patch_size - stride) // 2
    outLF = subLFs[:, :, :, :, :, bdr:bdr+stride, bdr:bdr+stride]
    outLF = rearrange(outLF, 'n1 n2 u v c h w -> u v c (n1 h) (n2 w)')

    return outLF

def crop_boundary(I, crop_size):
    '''crop the boundary (the last 2 dimensions) of a tensor'''
    if crop_size == 0:
        return I

    if crop_size > 0:
        size = list(I.shape)
        I_crop = I.view(-1, size[-2], size[-1])
        I_crop = I_crop[:, crop_size:-crop_size, crop_size:-crop_size]
        size[-1] -= crop_size * 2
        size[-2] -= crop_size * 2
        I_crop = I_crop.view(size)
        return I_crop