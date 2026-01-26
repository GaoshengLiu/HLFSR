import time
import argparse
import scipy.misc
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils import *
from model_att import Net
from tqdm import tqdm
from einops import rearrange
import scipy.io as sio
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
    parser.add_argument("--scale", type=int, default=4, help="upscale factor")
    parser.add_argument('--validset_dir', type=str, default='./Data/test_DLFD_x4/')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--crop', type=bool, default=True, help='Cropping into patches when validating')
    parser.add_argument("--patchsize_test", type=int, default=32, help="patchsize of LR images for inference")
    parser.add_argument("--minibatch_test", type=int, default=6, help="size of minibatch for inference")

    parser.add_argument('--save_array', type=bool, default=False, help='save LF as an array')
    parser.add_argument('--model_path', type=str, default='./log_crossatt_v4/HySR_5x5xSR_5x5_epoch_60.pth.tar')
    parser.add_argument('--save_path', type=str, default='./Results/')

    return parser.parse_args()

def test(cfg):
    net = Net(cfg.angRes, cfg.scale)
    net.to(cfg.device)
    cudnn.benchmark = True

    if os.path.isfile(cfg.model_path):
        model = torch.load(cfg.model_path, map_location={'cuda:1': cfg.device})
        net.load_state_dict(model['state_dict'])
    else:
        print("=> no model found at '{}'".format(cfg.model_path))
        
    test_set = TestSetLoader(cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=cfg.num_workers, batch_size=1, shuffle=False)
    if not (os.path.exists(cfg.save_path)):
        os.makedirs(cfg.save_path)

    with torch.no_grad():
        psnr_testset = []
        ssim_testset = []
        psnr_epoch_test, ssim_epoch_test = inference(test_loader, net)
        psnr_testset.append(psnr_epoch_test)
        ssim_testset.append(ssim_epoch_test)
        print(time.ctime()[4:-5] + ' Valid----%15s, PSNR---%f, SSIM---%f' % ('HCI_x'+str(cfg.scale), psnr_epoch_test, ssim_epoch_test))
        pass


def inference(test_loader, net):
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (lr, hr, label, lr_ycbcr_up) in (enumerate(test_loader)):
        seq_name = os.listdir(cfg.validset_dir)[idx_iter]
        
        if cfg.crop == False:
            with torch.no_grad():
                outLF = net(lr.to(cfg.device), hr.to(cfg.device))
                outLF = outLF.squeeze()
        else:
            patch_size = cfg.patchsize_test
            data, hr_cv = lr.squeeze(0), hr.reshape(1, 1, 1, hr.shape[-2], hr.shape[-1])
            sub_lfs = LFdivide(data,  patch_size, patch_size // 2)
            sub_hrs = LFdivide(hr_cv, patch_size*cfg.scale, patch_size*cfg.scale // 2)

            n1, n2, u, v, c, h, w = sub_lfs.shape
            sub_lfs =  rearrange(sub_lfs, 'n1 n2 u v c h w -> (n1 n2) u v c h w')
            sub_hrs =  rearrange(sub_hrs, 'n1 n2 u v c h w -> (n1 n2) u v c h w')
            #print(sub_lfs.shape)
            #print(sub_hrs.shape)
            mini_batch = cfg.minibatch_test
            num_inference = (n1 * n2) // mini_batch
            with torch.no_grad():
                out_lfs = []
                for idx_inference in range(num_inference):
                    torch.cuda.empty_cache()
                    input_lfs   = sub_lfs[idx_inference * mini_batch : (idx_inference+1) * mini_batch, :, :, :, :, :]
                    input_cvhrs = sub_hrs[idx_inference * mini_batch : (idx_inference+1) * mini_batch, :, :, :, :, :].squeeze(1).squeeze(2)
                    lf_out = net(input_lfs.to(cfg.device), input_cvhrs.to(cfg.device))
                    out_lfs.append(lf_out)
                if (n1 * n2) % mini_batch:
                    torch.cuda.empty_cache()
                    input_lfs   = sub_lfs[(idx_inference+1) * mini_batch :, :, :, :, :, :]
                    input_cvhrs = sub_hrs[(idx_inference+1) * mini_batch :, :, :, :, :, :].squeeze(1).squeeze(2)
                    lf_out = net(input_lfs.to(cfg.device), input_cvhrs.to(cfg.device))
                    out_lfs.append(lf_out)

            out_lfs = torch.cat(out_lfs, dim=0)
            out_lfs = rearrange(out_lfs, '(n1 n2) u v c h w -> n1 n2 u v c h w', n1=n1, n2=n2)
            outLF = LFintegrate(out_lfs, patch_size * cfg.scale, patch_size * cfg.scale // 2)
            outLF = outLF[:, :, :, 0 : data.shape[3] * cfg.scale, 0 : data.shape[4] * cfg.scale].squeeze()

        psnr, ssim = cal_metrics(label.squeeze(), outLF)
        print('Valid----%15s, PSNR---%f, SSIM---%f' % (seq_name[:-3], psnr, ssim))
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)

        # if not cfg.save_array:
        #     save_path = os.path.join(cfg.save_path, seq_name[:-3])
        #     if not (os.path.exists(save_path)):
        #         os.makedirs(save_path)
        #     for uu in range(cfg.angRes):
        #         for vv in range(cfg.angRes):
        #             img_path = save_path + '/' + str(uu).rjust(2,'0') +'_'+ str(vv).rjust(2,'0') + '.png'
        #             up_ycbcr = lr_ycbcr_up.squeeze()[uu, vv, ...].permute(1,2,0)
        #             up_ycbcr[:,:,0] = outLF[uu, vv,...]
        #             img = ycbcr2rgb(up_ycbcr.cpu().numpy())
        #             img = (np.clip(img, 0, 1)*255).astype(np.uint8)
        #             imageio.imwrite(img_path, img)  
        # else:
        #     img_path = cfg.save_path + '/' + seq_name[:-3] + '.png'
        #     lr_ycbcr_up = lr_ycbcr_up.squeeze()
        #     lr_ycbcr_up[:,:,0,:,:] = outLF
        #     img_save = rearrange(lr_ycbcr_up, 'u v c h w -> (u h) (v w) c')
        #     img_save = ycbcr2rgb(img_save.cpu().numpy())
        #     #img_save = (np.clip(img_save.cpu().numpy(), 0, 1)*255).astype(np.uint8)
        #     img_save = (np.clip(img_save, 0, 1)*255).astype(np.uint8)
        #     imageio.imwrite(img_path, img_save)

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

    return psnr_epoch_test, ssim_epoch_test

def main(cfg):
    #test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(cfg)
    test(cfg)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
