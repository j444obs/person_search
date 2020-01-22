"""
Author: https://github.com/jwyang/faster-rcnn.pytorch.git
Description: Region proposal network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fast_rcnn.config import cfg
from rpn.anchor_target_layer import AnchorTargetLayer
from rpn.proposal_layer import ProposalLayer
from utils.net_utils import smooth_l1_loss


class RPN(nn.Module):
    """Region proposal network"""

    def __init__(self, input_depth):
        super(RPN, self).__init__()
        self.num_anchors = len(cfg.ANCHOR_SCALES) * len(cfg.ANCHOR_RATIOS)

        # Define the conv layer processing input feature map
        self.rpn_conv = nn.Conv2d(input_depth, 512, 3, 1, 1, bias=True)

        # Define bg/fg classifcation score layer, 9(anchors) * 2(bg/fg)
        self.rpn_cls_score = nn.Conv2d(512, self.num_anchors * 2, 1, 1, 0)

        # Define anchor box offset prediction layer, 9(anchors) * 4(coords)
        self.rpn_bbox_pred = nn.Conv2d(512, self.num_anchors * 4, 1, 1, 0)

        # Define proposal layer
        self.rpn_proposal = ProposalLayer()

        # Define anchor target layer
        self.rpn_anchor_target = AnchorTargetLayer()

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        x = x.view(x.size(0), d, -1, x.size(3))
        return x

    def forward(self, base_feat, im_info, gt_boxes):
        assert base_feat.size(0) == 1, 'Only single item batches are supported'

        # Return feature map after conv-relu layer
        rpn_conv = F.relu(self.rpn_conv(base_feat), inplace=True)

        # Get rpn classification score
        rpn_cls_score = self.rpn_cls_score(rpn_conv)

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.num_anchors * 2)

        # Get rpn offsets to the anchor boxes
        rpn_bbox_pred = self.rpn_bbox_pred(rpn_conv)

        # Proposal layer
        rois = self.rpn_proposal(rpn_cls_prob.data, rpn_bbox_pred.data, im_info)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # Generating training labels and compute the rpn loss
        if self.training:
            assert gt_boxes is not None

            rpn_data = self.rpn_anchor_target(rpn_cls_score.data, gt_boxes, im_info)

            # Compute classification loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(1, -1, 2)
            rpn_label = rpn_data[0].view(1, -1)

            rpn_keep = rpn_label.view(-1).ne(-1).nonzero().view(-1)
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1, 2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = rpn_label.long()
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            # Compute bbox regression loss
            # rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            # rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            # rpn_bbox_targets = Variable(rpn_bbox_targets)

            self.rpn_loss_box = smooth_l1_loss(rpn_bbox_pred,
                                               rpn_bbox_targets,
                                               rpn_bbox_inside_weights,
                                               rpn_bbox_outside_weights,
                                               sigma=3,
                                               dim=[1, 2, 3])

        return rois, self.rpn_loss_cls, self.rpn_loss_box
