import torch
import torch.nn as nn
import torch.nn.functional as functional
from einops import rearrange
import numpy as np
from torch.autograd import Variable
import math
from dcn.modules.deform_conv import DeformConv


class Net(nn.Module):
    def __init__(self, angRes, scale):
        super(Net, self).__init__()
        channel=64
        self.angRes = angRes
        self.scale = scale
        self.initFeaExt = nn.Sequential(
                    nn.Conv2d(1, channel, kernel_size=(3,3), stride=1, dilation=1, padding=(1,1), bias=False),
                    nn.LeakyReLU(0.1, inplace=True))
        pre_up = False
        if scale == 8:
            pre_up = True
            scale = 4

        self.deeFeaExt = CascadeAlterBlock(3, 8, channel, angRes, scale, pre_up)

        self.hr_FeaExt = nn.Sequential(
                    nn.Conv2d(1, channel, kernel_size=3, stride=1, dilation=1, padding=1, bias=False),
                    nn.LeakyReLU(0.1, inplace=True),
                    ResB(channel),
                    ResB(channel),
                    nn.Conv2d(channel, channel//2, kernel_size=3, stride=1, dilation=1, padding=1, bias=False))
        
        self.upsampling = nn.Sequential(
                    nn.Conv2d(channel, channel*scale ** 2, kernel_size=1, padding=0, dilation=1, bias=False),
                    nn.PixelShuffle(scale),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, lr, hr_c):
        b, u, v, c, h, w = lr.shape
        lr_in = rearrange(lr, 'b u v c h w -> (b u v) c h w')
        lr_bicup = functional.interpolate(lr_in, scale_factor=(self.scale, self.scale), mode='bicubic', align_corners=False)
        fea_hr = self.hr_FeaExt(hr_c)
        fea_lr = self.initFeaExt(lr_in)
        fea_lr = rearrange(fea_lr, '(b u v) c h w -> b (u v) c h w', u=u, v=v)
        
        fea_lr, _ = self.deeFeaExt(fea_lr, fea_hr)
        
        out = self.upsampling(rearrange(fea_lr, 'b (u v) c h w -> (b u v) c h w', u=u, v=v))
        out = rearrange(out+lr_bicup, '(b u v) c h w -> b u v c h w', u=u, v=v)
        return out

class SepFusion(nn.Module):
    def __init__(self, angRes, channels, scale, pre_up):
        super(SepFusion, self).__init__()
        self.angRes = angRes
        self.scale = scale
        deform_group=4
        self.pre_up = pre_up

        self.dcn = DeformConv(channels//2, channels//2, kernel_size=3, stride=1, padding=1, deformable_groups=deform_group)
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale)
        self.offset = nn.Sequential(
                    nn.Conv2d(channels+channels//2, channels, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(channels, 9*2*deform_group, kernel_size=1, stride=1, padding=0, bias=True))##
        if self.pre_up:
            self.upsampling = nn.Sequential(
                    nn.Conv2d(channels, channels*2 ** 2, kernel_size=1, padding=0, dilation=1, bias=False),
                    nn.PixelShuffle(2),
                    nn.LeakyReLU(0.1, inplace=True))
            self.out = nn.Conv2d(channels+channels//2*((scale//2)**2), channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=True)
        else:
            self.out = nn.Conv2d(channels+channels//2*(scale**2), channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=True)

    def forward(self, lf_lr, cv_hr):
        b,n,c,h,w = lf_lr.shape
        buffer = []
        for i in range(n):
            v_lr = lf_lr[:,i,:,:,:]
            v_lr_up = self.up(v_lr)
            offset  = self.offset(torch.cat((v_lr_up, cv_hr), dim=1))
            sampled = self.dcn(cv_hr, offset)
            if self.pre_up:
                sampled = rearrange(sampled, 'b c (h a1) (w a2) -> b (a1 a2 c) h w', a1=self.scale//2, a2=self.scale//2, h=h*2,w=w*2)
                v_lr    = self.upsampling(v_lr)
            else:
                sampled = rearrange(sampled, 'b c (h a1) (w a2) -> b (a1 a2 c) h w', a1=self.scale, a2=self.scale, h=h,w=w)
            buffer.append(self.out(torch.cat((v_lr, sampled), dim=1)))
        buffer = torch.stack(buffer, dim=1)
        return buffer

class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
        )
    def __call__(self,x):
        out = self.body(x)
        return out + x
class CascadeAlterBlock(nn.Module):
    def __init__(self, n_group, n_block, channels, angRes, scale, pre_up=False):
        super(CascadeAlterBlock, self).__init__()
        self.n_group = n_group
        Groups = []
        self.pre_up = pre_up
        if not self.pre_up:
            scale_1, scale_2 = 4, 4
        else:
            scale_1, scale_2 = 8, 4
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        for i in range(n_group):
            if i == 0:
                Groups.append(AlterBlock(n_block, channels, angRes, scale_1, pre_up=pre_up, last=False))
            elif 0 < i < (n_group-1):
                Groups.append(AlterBlock(n_block, channels, angRes, scale_2, last=False))
            else:
                Groups.append(AlterBlock(n_block, channels, angRes, scale_2, last=True))
        #print(len(Groups))
        self.Group = nn.Sequential(*Groups)
        self.convlr = nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 1, bias=False)

    def forward(self, x_lr, cv_hr):
        b, n, c, h, w = x_lr.shape           
        buffer_l = x_lr
        buffer_h = cv_hr        
        for i in range(self.n_group):
            buffer_l, buffer_h = self.Group[i](buffer_l, buffer_h)
        if self.pre_up:
            buffer = self.convlr(buffer_l.contiguous().view(b*n, c, h*2, w*2))
            x_res = self.up(x_lr.contiguous().view(b*n, c, x_lr.shape[-2], x_lr.shape[-1]))
            out = (buffer + x_res).contiguous().view(b, n, c, h*2, w*2)
        else:
            buffer = self.convlr(buffer_l.contiguous().view(b*n, c, h, w))
            out = buffer.contiguous().view(b,n, c, h, w) + x_lr

        return out, buffer_h
class AlterBlock(nn.Module):
    def __init__(self, n_block, channels, angRes, scale, pre_up=False, last=False):
        super(AlterBlock, self).__init__()
        self.n_block = n_block
        self.angRes = angRes
        self.pre_up = pre_up
        self.last = last 
        Blocks = []
        for i in range(n_block):
            Blocks.append(SpaAngConv(angRes, channels))
        self.block = nn.Sequential(*Blocks)
        self.sepFusin = SepFusion(angRes, channels, scale, pre_up) 
        if self.pre_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        if not last:    
            self.conv_hr = nn.Sequential(
                nn.Conv2d(channels//2, channels//2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(0.1, True))
    def forward(self, x_lr, cv_hr):
        #b, n, c, h, w = x_lr.shape    
        buffer_hr = cv_hr
        buffer = self.sepFusin(x_lr, buffer_hr)
        b, n, c, h, w = buffer.shape 
        buffer = buffer.reshape(b, self.angRes, self.angRes, c, h, w)
        for i in range(self.n_block):
            buffer = self.block[i](buffer)        
        buffer = self.conv(buffer.contiguous().view(b*n, c, h, w))
        
        if self.pre_up:
            x_lr = self.up(x_lr.contiguous().view(b*n, c, x_lr.shape[-2], x_lr.shape[-1])).contiguous().view(b, n, c, h, w)
        buffer = buffer.contiguous().view(b, n, c, h, w) + x_lr
        if not self.last:
            buffer_hr = self.conv_hr(buffer_hr) + cv_hr
        return buffer, buffer_hr
class SpaAngConv(nn.Module):
    def __init__(self, angRes, channels):
        super(SpaAngConv, self).__init__()
        self.spa_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.ang_conv = nn.Conv2d(channels, channels//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.att_0 = CrossAtt(channels//2, channels)
        self.att_1 = CrossAtt(channels, channels//2)
        self.fuse = nn.Conv2d(channels+channels//2, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.lrelu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        b, u, v, c, h, w = x.shape
        buffer = x
        buffer_s = rearrange(x, 'b u v c h w -> (b u v) c h w')
        buffer_s = self.lrelu(self.spa_conv(buffer_s))
          
        buffer_a = rearrange(x, 'b u v c h w -> (b h w) c u v')
        buffer_a = self.lrelu(self.ang_conv(buffer_a))
        buffer_a = rearrange(buffer_a, '(b h w) c u v -> (b u v) c h w', b=b, h=h, w=w)
        
        buffer_sout = buffer_s*(self.att_0(buffer_a)+1)
        buffer_aout = buffer_a*(self.att_1(buffer_s)+1)
        
        buffer = self.fuse(torch.cat((buffer_sout, buffer_aout), dim=1))
    
        return buffer.reshape(b,u,v,c,h,w) + x        
class CrossAtt(nn.Module):
    def __init__(self, ch1, ch2):
        super(CrossAtt, self).__init__()
        self.outch = ch2
        self.conv = nn.Sequential(
                        nn.Conv2d(in_channels = ch1, out_channels = ch2, kernel_size = 1, stride = 1, padding = 0, dilation=1),
                        nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.att_c = nn.Conv2d(ch2, ch2, 1, 1, 0)

    def forward(self, x1):
        b, _, h, w = x1.shape
        buffer = x1
        buffer = buffer.contiguous().view(b, -1, h, w)

        buffer = self.conv(buffer)    
        out = self.att_c(buffer)
        out = out.contiguous().view(b, self.outch, h, w)
        return out
    
if __name__ == "__main__":
    angRes = 5
    scale = 4
    net = Net(angRes, scale).cuda()
    from thop import profile
    lr = torch.randn(1, angRes, angRes, 1, 32, 32).cuda()
    hr = torch.randn(1,1,128,128).cuda()

    flops, params = profile(net, inputs=(lr,hr))
    print('   Number of parameters: %.2fM' % (params / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops / 1e9))