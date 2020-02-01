"""
Author: 520Chris
Description: Train a person search network.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.optim as optim

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

    # print('Called with args:')
    # print(args)

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
    # print('%s roidb entries' % len(roidb))

    output_dir = get_output_dir(imdb.name)
    print('Output will be saved to `%s`' % output_dir)

    dataloader = DataLoader(roidb)
    net = Network()
    optimizer = optim.SGD(net.get_training_params(),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40000, gamma=0.1)

    iter_size = 2  # accumulated gradient update
    display = 20
    loss_total = 0
    for i in range(args.max_iters):
        blob = dataloader.get_next_minibatch()
        output = net(torch.from_numpy(blob['data']).cuda(),
                     torch.from_numpy(blob['im_info']).cuda(),
                     torch.from_numpy(blob['gt_boxes']).cuda())
        cls_prob, bbox_pred, feat, rpn_loss_cls, rpn_loss_bbox, loss_cls, loss_bbox, loss_id = output
        loss = (rpn_loss_cls + rpn_loss_bbox + loss_cls + loss_bbox + loss_id) / iter_size
        loss_total += loss
        loss.backward()

        if (i + 1) % iter_size == 0:
            optimizer.step()
            optimizer.zero_grad()

        if (i + 1) % display == 0:
            print("Iteration [%s] / [%s]: loss: %.4f" % (i + 1, args.max_iters, loss_total / display))
            print("rpn_loss_cls: %.4f, rpn_loss_bbox: %.4f, loss_cls: %.4f, loss_bbox: %.4f, loss_id: %.4f" %
                  (rpn_loss_cls, rpn_loss_bbox, loss_cls, loss_bbox, loss_id))
            loss_total = 0

        # adjust learning rate
        scheduler.step()
