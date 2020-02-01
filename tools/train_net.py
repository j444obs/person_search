"""
Author: 520Chris
Description: Train a person search network
"""

import argparse
import os
import random

import numpy as np
import torch

from datasets.factory import get_imdb
from models.network import Network
from roi_data_layer.dataloader import DataLoader
from utils.config import cfg, cfg_from_file, get_output_dir


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a person search network')
    parser.add_argument('--gpu', dest='gpu',
                        help='GPU device id to use [0,1,2,3,4,5,6,7,8]',
                        default='0', type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--snapshot', dest='previous_state',
                        help='initialize with previous solver state',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    return parser.parse_args()


def prepare_imdb(name):
    print("Loading image database: %s" % name)
    imdb = get_imdb(name)
    print("Done.")
    if cfg.TRAIN.USE_FLIPPED:
        print('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images()
        print('Done.')
    return imdb


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if not args.randomize:
        # Fix the random seeds (numpy and pytorch) for reproducibility
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed_all(cfg.RNG_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(cfg.RNG_SEED)
        random.seed(cfg.RNG_SEED)
        os.environ['PYTHONHASHSEED'] = str(cfg.RNG_SEED)

    imdb = prepare_imdb(args.imdb_name)
    roidb = imdb.roidb
    print('%s roidb entries' % len(roidb))

    output_dir = get_output_dir(imdb.name)
    print('Output will be saved to `%s`' % output_dir)

    dataloader = DataLoader(roidb)
    net = Network()
    net.train()
    for i in range(1000):
        blob = dataloader.get_next_minibatch()
        res = net(torch.from_numpy(blob['data']),
                  torch.from_numpy(blob['im_info']),
                  torch.from_numpy(blob['gt_boxes']))
