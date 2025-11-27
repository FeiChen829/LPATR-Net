import os
import torch
import argparse
from torch.backends import cudnn
from eval import _eval
from models.LPATRNet import build_net
from train import _train
import warnings
warnings.filterwarnings("ignore")


def main(args):
    # CUDNN
    cudnn.benchmark = True

    model = build_net()

    if torch.cuda.is_available():
        model.cuda()
    if args.mode == 'train':
        _train(model, args)
    if args.mode == 'test':
        _eval(model, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='LPATRNet')
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epoch', type=int, default=1000)
    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--data_dir', type=str, default=r'./datasets/reside-indoor')
    parser.add_argument('--resume', type=str, default='',)
    parser.add_argument('--DatasetName', type=str, default='reside-outdoor',choices=['reside-indoor','reside-outdoor','O-HAZE'])
    parser.add_argument('--pre_model_path', type=str, default=r'Outdoor_Best_41.24_0.9972.pkl')
    parser.add_argument('--input_file_path', type=str, default='./test_single_images/input/test.jpg')
    parser.add_argument('--output_file_path', type=str, default='./test_single_images/output')
    args = parser.parse_args()
    args.log_dir = os.path.join('results', args.model_name, args.DatasetName, 'log')
    args.model_save_dir = os.path.join('results', args.model_name, args.DatasetName, 'checkpoint')

    args.result_save_dir = os.path.join('results', args.model_name, args.DatasetName, 'output')

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_save_dir):
        os.makedirs(args.result_save_dir)
    print(args)
    main(args)
