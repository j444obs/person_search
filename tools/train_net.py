import argparse
import os
import os.path as osp
import random
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.psdb import PSDB
from datasets.sampler import PSSampler
from models.network import Network
from utils.config import cfg, cfg_from_file


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a person search network.')
    parser.add_argument('--gpu', default=-1, type=int,
                        help='GPU device id to use. Default: -1, means using CPU')
    parser.add_argument('--epoch', default=5, type=int,
                        help='Number of epochs to train. Default: 5')
    parser.add_argument('--weights', default=None, type=str,
                        help='Initialize with pretrained model weights. Default: None')
    parser.add_argument('--checkpoint', default=None, type=str,
                        help='Initialize with previous solver state. Default: None')
    parser.add_argument('--cfg', default=None, type=str,
                        help='Optional config file. Default: None')
    parser.add_argument('--data_dir', default=None, type=str,
                        help='The directory that saving experimental data. Default: None')
    parser.add_argument('--dataset', default='psdb_train', type=str,
                        help='Dataset to train on. Default: psdb_train')
    parser.add_argument('--rand', action='store_true',
                        help='Do not use a fixed seed. Default: False')
    parser.add_argument('--tbX', action='store_true',
                        help='Enable tensorboardX. Default: False')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg:
        cfg_from_file(args.cfg)
    if args.data_dir:
        cfg.DATA_DIR = args.data_dir

    if not args.rand:
        # Fix the random seeds (numpy and pytorch) for reproducibility
        print("Set to none random mode.")
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed_all(cfg.RNG_SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(cfg.RNG_SEED)
        np.random.seed(cfg.RNG_SEED)
        os.environ['PYTHONHASHSEED'] = str(cfg.RNG_SEED)

    output_dir = osp.abspath(osp.join(cfg.DATA_DIR, 'trained_model'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    assert args.dataset in ['psdb_train', 'psdb_test'], "Unknown dataset: %s" % args.dataset
    psdb = PSDB(args.dataset)
    dataloader = DataLoader(psdb, batch_size=1, sampler=PSSampler(psdb))
    print("Loaded dataset: %s" % args.dataset)

    # Set model and optimizer
    if args.weights is None:
        args.weights = osp.abspath(osp.join(cfg.DATA_DIR, 'pretrained_model', 'resnet50_caffe.pth'))
    net = Network(args.weights)

    lr = cfg.TRAIN.LEARNING_RATE
    weight_decay = cfg.TRAIN.WEIGHT_DECAY
    params = []
    for k, v in net.named_parameters():
        if v.requires_grad:
            if 'BN' in k:
                params += [{'params': [v], 'lr': lr, 'weight_decay': 0}]
            elif 'bias' in k:
                params += [{'params': [v], 'lr': 2 * lr, 'weight_decay': 0}]
            else:
                params += [{'params': [v], 'lr': lr, 'weight_decay': weight_decay}]
    optimizer = optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    # Training settings
    start_epoch = 0
    display = 20  # Display the loss every `display` steps
    lr_decay = 4  # Decay the learning rate every `lr_decay` epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40000, gamma=0.1)
    iter_size = 2  # Each update use accumulated gradient by `iter_size` iterations
    use_caffe_smooth_loss = True
    average_loss = 100  # Be used to calculate smoothed loss

    # Load checkpoint
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("Loaded checkpoint from: %s" % args.checkpoint)

    # Place the model on cuda device
    if args.gpu != -1:
        net.cuda(args.gpu)

    # Use tensorboardX to visualize experimental results
    if args.tbX:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter("logs")

    net.train()
    start = time.time()
    accumulated_step = 0
    loss = 0
    losses = []
    ave_loss = 0
    smoothed_loss = 0
    for epoch in range(start_epoch, args.epoch):
        # if epoch % lr_decay == 0:
        #     # Adjust learning rate
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = 0.1 * param_group['lr']

        for step, input_data in enumerate(dataloader):
            data, im_info, gt_boxes = input_data[0][0], input_data[1][0], input_data[2][0]
            if args.gpu != -1:
                data = data.cuda(args.gpu)
                im_info = im_info.cuda(args.gpu)
                gt_boxes = gt_boxes.cuda(args.gpu)

            real_step = int(step / iter_size)
            _, _, _, rpn_loss_cls, rpn_loss_bbox, loss_cls, loss_bbox, loss_id = net(data, im_info, gt_boxes)
            loss_iter = (rpn_loss_cls + rpn_loss_bbox + loss_cls + loss_bbox + loss_id) / iter_size
            loss += loss_iter
            loss_iter.backward()
            accumulated_step += 1

            if accumulated_step == iter_size:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  # Adjust learning rate every real step

                if use_caffe_smooth_loss:
                    if len(losses) < average_loss:
                        losses.append(loss)
                        size = len(losses)
                        smoothed_loss = (smoothed_loss * (size - 1) + loss) / size
                    else:
                        idx = real_step % average_loss
                        smoothed_loss += (loss - losses[idx]) / average_loss
                        losses[idx] = loss
                else:
                    ave_loss += loss

                loss = 0
                accumulated_step = 0

                if real_step % display == 0:
                    if use_caffe_smooth_loss:
                        display_loss = smoothed_loss
                    else:
                        display_loss = ave_loss / display if real_step > 0 else ave_loss
                        ave_loss = 0

                    real_steps_per_epoch = int(len(dataloader) / iter_size)
                    print("-----------------------------------------------------------------")
                    print("Epoch: [%s / %s], iteration [%s / %s], loss: %.4f" %
                          (epoch, args.epoch - 1, real_step, real_steps_per_epoch - 1, display_loss))
                    print("Time cost: %.2f seconds" % (time.time() - start))
                    print("Learning rate: %s" % optimizer.param_groups[0]['lr'])
                    print("The %s-th iteration loss:" % real_step)
                    print("  rpn_loss_cls: %.4f, rpn_loss_bbox: %.4f" % (rpn_loss_cls, rpn_loss_bbox))
                    print("  loss_cls: %.4f, loss_bbox: %.4f, loss_id: %.4f" % (loss_cls, loss_bbox, loss_id))

                    start = time.time()

                    if args.tbX:
                        log_info = {
                            'loss': display_loss,
                            'rpn_loss_cls': rpn_loss_cls,
                            'rpn_loss_bbox': rpn_loss_bbox,
                            'loss_cls': loss_cls,
                            'loss_bbox': loss_bbox,
                            'loss_id': loss_id
                        }
                        logger.add_scalars("Train/Loss", log_info, epoch * real_steps_per_epoch + real_step)

        # Save checkpoint every epoch
        save_name = os.path.join(output_dir, 'resnet50_epoch_%s.pth' % epoch)
        torch.save({
            'epoch': epoch,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, save_name)

    if args.tbX:
        logger.close()
