import time
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import *
from tqdm import tqdm
from model_w_att import Net
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# Settings
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--parallel', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=64)
    parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--model_name', type=str, default='HySR')
    parser.add_argument('--trainset_dir', type=str, default='./Data/train_data/')
    parser.add_argument('--validset_dir', type=str, default='./Data/test_HCI_x4/')
    parser.add_argument('--patchsize', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--n_epochs', type=int, default=60, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=16, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.6, help='learning rate decaying factor')
    parser.add_argument('--crop', type=bool, default=True, help='Cropping into patches when validating')
    parser.add_argument("--patchsize_test", type=int, default=32, help="patchsize of LR images for inference")
    parser.add_argument("--minibatch_test", type=int, default=1, help="size of minibatch for inference")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='./log/HySR_5x5xSR_5x5_epoch_1.pth.tar')

    return parser.parse_args()
if not os.path.exists('./log'):
    os.mkdir('./log')
def train(cfg):
    setup_seed(10)
    if cfg.parallel:
        cfg.device = 'cuda:0'
    net = Net(cfg.angRes, cfg.scale)
    net.to(cfg.device)
    cudnn.benchmark = True
    epoch_state = 0
    criterion_Loss = torch.nn.L1Loss().to(cfg.device)
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)

    if cfg.load_pretrain:
        if os.path.isfile(cfg.model_path):
            model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
            net.load_state_dict(model['state_dict'], strict=False)
            optimizer.load_state_dict(model['optimazer'])
            epoch_state = model["epoch"]
            print("load pre-train at epoch {}".format(epoch_state))
        else:
            print("=> no model found at '{}'".format(cfg.model_path))

    if cfg.parallel:
        net = torch.nn.DataParallel(net, device_ids=[0, 1])
    scheduler._step_count = epoch_state

    val_set = ValSetLoader(cfg)
    val_loader = DataLoader(dataset=val_set, num_workers=cfg.num_workers, batch_size=1, shuffle=False)
    train_set = TrainSetLoader(cfg)
    train_loader = DataLoader(dataset=train_set, num_workers=cfg.num_workers, batch_size=cfg.batch_size, shuffle=True)

    loss_list = []
    for idx_epoch in range(epoch_state, cfg.n_epochs):        
        loss_epoch = []
        for idx_iter, (lr, hr, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
            lr, hr, label = lr.to(cfg.device), hr.to(cfg.device), label.to(cfg.device)
            #with torch.autograd.detect_anomaly():
            out = net(lr, hr)
            loss = criterion_Loss(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.data.cpu())

        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch----%5d, loss---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean())))
            txtfile = open('./log/' + cfg.model_name + '_training.txt', 'a')
            txtfile.write(time.ctime()[4:-5] + ' Epoch----%5d, loss---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean())) + '\n')
            txtfile.close()
            if cfg.parallel:
                save_ckpt({
                    'epoch': idx_epoch + 1,
                    'optimazer': optimizer.state_dict(),
                    'state_dict': net.module.state_dict(),  # for torch.nn.DataParallel
                    'loss': loss_list,},
                    save_path='./log/', filename=cfg.model_name + '_' + str(cfg.angRes) + 'x' + str(cfg.angRes)+ 'xSR_' + str(cfg.angRes) +
                                'x' + str(cfg.angRes) + '_epoch_' + str(idx_epoch + 1) + '.pth.tar')
            else:
                save_ckpt({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.state_dict(),
                    'optimazer': optimizer.state_dict(),
                    'loss': loss_list,},
                    save_path='./log/', filename=cfg.model_name + '_' + str(cfg.angRes) + 'x' + str(cfg.angRes)+ 'xSR_' + str(cfg.angRes) +
                                'x' + str(cfg.angRes) + '_epoch_' + str(idx_epoch + 1) + '.pth.tar')
            loss_epoch = []

        ''' evaluation '''
        with torch.no_grad():
            psnr_testset = []
            ssim_testset = []
            psnr_epoch_test, ssim_epoch_test = valid(val_loader, net)
            psnr_testset.append(psnr_epoch_test)
            ssim_testset.append(ssim_epoch_test)
            print(time.ctime()[4:-5] + ' Valid----%15s, PSNR---%f, SSIM---%f' % ('HCI_x'+str(cfg.scale), psnr_epoch_test, ssim_epoch_test))
            txtfile = open('./log/' + cfg.model_name + '_training.txt', 'a')
            txtfile.write('Dataset----%10s,\t PSNR---%f,\t SSIM---%f\n' % ('HCI_x'+str(cfg.scale), psnr_epoch_test, ssim_epoch_test))
            txtfile.close()
            pass


        scheduler.step()


def valid(test_loader, net):
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (lr, hr, label) in (enumerate(test_loader)):

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
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

    return psnr_epoch_test, ssim_epoch_test


def save_ckpt(state, save_path='./log', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename), _use_new_zipfile_serialization=False)
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.enabled = False

if __name__ == '__main__':
    cfg = parse_args()
    train(cfg)
