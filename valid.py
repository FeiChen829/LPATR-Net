import torch
from data import valid_dataloader
from utils import Adder
import os
from pytorch_msssim import ssim
import torch.nn.functional as f


def _valid(model, args, ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataloader
    ots = valid_dataloader(args.data_dir, batch_size=1, num_workers=0,DatasetName=args.DatasetName)
    model.eval()

    # Adder
    psnr_adder = Adder()
    ssim_adder = Adder()

    # pad factor
    factor = 8

    with torch.no_grad():
        print('==========================================================')
        print('===> Start Evaluation <===')
        print(f'==> epoch {ep} _ ')
        for idx, data in enumerate(ots):
            input_img, label_img = data
            input_img, label_img = input_img.to(device), label_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor * factor)
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')

            pred = model(input_img)[2]
            pred = pred[:, :, :h, :w]

            pred_clip = torch.clamp(pred, 0, 1)

            psnr_val = 10 * torch.log10(1 / f.mse_loss(pred_clip, label_img))
            down_ratio = max(1, round(min(H, W) / 256))
            ssim_val = ssim(f.adaptive_avg_pool2d(pred_clip, (int(H / down_ratio), int(W / down_ratio))),
                            f.adaptive_avg_pool2d(label_img, (int(H / down_ratio), int(W / down_ratio))),
                            data_range=1, size_average=False)
            ssim_adder(ssim_val)
            psnr_adder(psnr_val)

        print('     The average PSNR is %.2f dB' % (psnr_adder.average()))
        print('     The average SSIM is %.4f dB' % (ssim_adder.average()))
        print('==========================================================')
    model.train()
    return psnr_adder.average(), ssim_adder.average()
