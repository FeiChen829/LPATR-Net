import os
import torch
from torchvision.utils import save_image
import argparse
from models.LPATRNet import build_net
from PIL import Image as Image
from torchvision.transforms import functional as F
import torchvision.transforms as transforms

torch.set_printoptions(profile="full")
import torch.nn.functional as f


def test_sigle_image(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load
    model = build_net().to(device)
    state_dict = torch.load(args.pre_model_path)
    model.load_state_dict(state_dict['model'])
    model.eval()

    # open
    input_img = Image.open(args.input_file_path).convert('RGB')
    input_img = F.to_tensor(input_img).unsqueeze(0).to(device)

    with torch.no_grad():
        # pad
        factor = 8
        h, w = input_img.shape[2], input_img.shape[3]
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor * factor)
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')
        # run
        outputs = model(input_img)
        out = outputs[2][:, :, :h, :w]
        # save
        file_name = os.path.basename(args.input_file_path)
        save_path = os.path.join(args.output_file_path, file_name)
        save_image(out, save_path)
        print('Successful processing!',f'result is saved at {save_path}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--pre_model_path', type=str, default='Outdoor_Best_41.24_0.9972.pkl')
    parser.add_argument('--input_file_path', type=str, default='./test_single_images/input/test.jpg')
    parser.add_argument('--output_file_path', type=str, default='./test_single_images/output')

    args = parser.parse_args()

    if not os.path.exists(args.output_file_path):
        os.makedirs(args.output_file_path)

    # model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # test_sigle_image
    test_sigle_image(args)
