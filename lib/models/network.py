"""
Author: 520Chris
Description: person search network based on resnet50.
"""

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign, RoIPool

from models.base_feat_layer import BaseFeatLayer
from models.proposal_feat_layer import ProposalFeatLayer
from roi_data_layer.dataloader import DataLoader
from rpn.proposal_target_layer import ProposalTargetLayer
from rpn.rpn_layer import RPN
from utils.config import cfg, cfg_from_file, get_output_dir


class Network(nn.Module):
    """Person search network."""

    def __init__(self):
        super(Network, self).__init__()
        rpn_depth = 1024  # depth of the feature map fed into RPN
        num_classes = 2  # bg and fg

        # Extracting feature layer
        self.base_feat_layer = BaseFeatLayer()
        self.proposal_feat_layer = ProposalFeatLayer()

        # RPN
        self.rpn = RPN(rpn_depth)
        self.proposal_target_layer = ProposalTargetLayer(num_classes=num_classes)
        self.rois = None  # proposals produced by RPN

        # Pooling layer
        pool_size = 14
        self.roi_align = RoIAlign((pool_size, pool_size), 1.0 / 16.0, 0)
        self.roi_pool = RoIPool((pool_size, pool_size), 1.0 / 16.0)

        # Identification layer
        self.det_score = nn.Linear(2048, 2)
        self.bbox_pred = nn.Linear(2048, num_classes * 4)
        self.feat_lowdim = nn.Linear(2048, 256)

    def forward(self, im_data, im_info, gt_boxes, is_prob=False, rois=None):
        assert im_data.size(0) == 1, 'Single batch only.'

        # Extract basic feature from image data
        base_feat = self.base_feat_layer(im_data)

        if not is_prob:
            # Feed base feature map to RPN to obtain rois
            self.rois, rpn_loss_cls, rpn_loss_box = self.rpn(base_feat, im_info, gt_boxes)
        else:
            assert rois is not None, "rois is not given in detect probe mode."
            self.rois, rpn_loss_cls, rpn_loss_box = rois, 0, 0

        # If it is training phase, then use ground truth bboxes for refining
        if self.training:
            roi_data = self.proposal_target_layer(self.rois, gt_boxes)
            self.rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws, aux_label = roi_data
        else:
            rois_label, rois_target, rois_inside_ws, rois_outside_ws, aux_label = [None] * 5

        # Do roi pooling based on region proposals
        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.roi_align(base_feat, self.rois)
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.roi_pool(base_feat, self.rois)
        else:
            raise NotImplementedError("Only support roi_align and roi_pool.")

        # Extract the features of proposals
        proposal_feat = self.proposal_feat_layer(pooled_feat).squeeze()

        det_score = self.det_score(proposal_feat)
        bbox_pred = self.bbox_pred(proposal_feat)
        feat_lowdim = F.normalize(self.feat_lowdim(proposal_feat))

        return det_score, bbox_pred, feat_lowdim

    # def init_from_caffe(self):
    #     dict_new = self.state_dict().copy()
    #     caffe_weights = pickle.load(open('caffe_model_weights.pkl', "rb"), encoding='latin1')
    #     for k in self.state_dict():
    #         frags = k.split('.')

    #         # Layer name mapping
    #         if frags[-2] == 'rpn_conv':
    #             name = 'rpn_conv/3x3'
    #         if frags[-2] in ['id_det_score', 'id_bbox_pred', 'id_feat_lowdim']:
    #             name = frags[-2][3:]
    #         elif frags[-2] in ['rpn_cls_score', 'rpn_bbox_pred']:
    #             name = frags[-2]
    #         else:
    #             name = 'caffe.' + frags[-2]

    #         if name not in caffe_weights:
    #             print("Layer: %s not found" % k)
    #             continue

    #         if frags[-1] == 'weight':
    #             dict_new[k] = torch.from_numpy(caffe_weights[name][0]).reshape(dict_new[k].shape)
    #         else:
    #             dict_new[k] = torch.from_numpy(caffe_weights[name][1]).reshape(dict_new[k].shape)

    #     net.load_state_dict(dict_new)
    #     print("load caffe model successfully!")
