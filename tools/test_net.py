import argparse
import os.path as osp
import pickle

import torch

from datasets.psdb import PSDB
from evaluate import evaluate_detections, evaluate_search
from models.network import Network
from test_gallery import detect_and_exfeat
from test_probe import exfeat
from utils import pickle, unpickle
from utils.config import cfg_from_file


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Test the person search network.')
    parser.add_argument('--gpu', default=-1, type=int,
                        help='GPU device id to use. Default: -1, means using CPU')
    parser.add_argument('--checkpoint', default=None, type=str,
                        help='The checkpoint to be tested. Default: None')
    parser.add_argument('--cfg', default=None, type=str,
                        help='Optional config file. Default: None')
    parser.add_argument('--dataset', default='psdb_test', type=str,
                        help='Dataset to test on. Default: psdb_test')
    parser.add_argument('--eval_only', action='store_true',
                        help='Evaluation with pre extracted features. Default: False')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg:
        cfg_from_file(args.cfg)
    if args.checkpoint is None:
        raise KeyError("--checkpoint option must be specified.")

    psdb = PSDB(args.dataset)
    net = Network()
    checkpoint = torch.load(osp.abspath(args.checkpoint))
    net.load_state_dict(checkpoint["model"])
    net.eval()
    if args.gpu != -1:
        net.cuda(args.gpu)

    if args.eval_only:
        gboxes = unpickle('gallery_detections.pkl')
        gfeatures = unpickle('gallery_features.pkl')
        pfeatures = unpickle('probe_features.pkl')
    else:
        # 1. Detect and extract features from all the gallery images in the imdb
        gboxes, gfeatures = detect_and_exfeat(net, psdb)

        # 2. Only extract features from given probe rois
        pfeatures = exfeat(net, psdb.probes)

        pickle(gboxes, 'gallery_detections.pkl')
        pickle(gfeatures, 'gallery_features.pkl')
        pickle(pfeatures, 'probe_features.pkl')

    # Evaluate
    evaluate_detections(psdb, gboxes, det_thresh=0.5)
    evaluate_detections(psdb, gboxes, det_thresh=0.5, labeled_only=True)
    evaluate_search(psdb, gboxes, gfeatures['feat'], pfeatures['feat'], det_thresh=0.5, gallery_size=100)
