import os
import torch
from torchvision.transforms import functional as F
from torchvision.utils import save_image
import argparse

from models.LPATRNet import build_net
from utils import Adder
from data import test_dataloader
from skimage.metrics import peak_signal_noise_ratio
import time
from pytorch_msssim import ssim
import torch.nn.functional as f
import numpy as np

torch.set_printoptions(profile="full")



def _eval(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0,DatasetName=args.DatasetName)
    # load
    state_dict = torch.load(args.pre_model_path)
    model.load_state_dict(state_dict['model'])
    model.eval()

    factor = 8
    # adder
    psnr_adder = Adder()
    ssim_adder = Adder()
    with torch.no_grad():
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor * factor)
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')
            outputs = model(input_img)
            out = outputs[2][:, :, :h, :w]

            pred_clip = torch.clamp(out, 0, 1)
            psnr_val = 10 * torch.log10(1 / f.mse_loss(pred_clip, label_img))
            down_ratio = max(1, round(min(H, W) / 256))
            ssim_val = ssim(f.adaptive_avg_pool2d(pred_clip, (int(H / down_ratio), int(W / down_ratio))),
                            f.adaptive_avg_pool2d(label_img, (int(H / down_ratio), int(W / down_ratio))),
                            data_range=1, size_average=False)
            print(iter_idx+1,'|',name[0],'|','PSNR',psnr_val.item(),'|','SSIM',ssim_val.item())
            ssim_adder(ssim_val)
            psnr_adder(psnr_val)

            save_image(out, os.path.join(args.result_save_dir, name[0]))

        print('     The average PSNR is %.2f dB' % (psnr_adder.average()))
        print('     The average SSIM is %.4f dB' % (ssim_adder.average()))
        print('==========================================================')


if __name__ == '__main__':
    # model
    model = build_net().to('cuda')
    state_dict = torch.load('')
    model.load_state_dict(state_dict['model'])
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.result_save_dir = r"result"

    if not os.path.exists(args.result_save_dir):
        os.makedirs(args.result_save_dir)
    _eval(model, args)

