# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os

import torch
import torch.nn as nn

from rpn.generate_anchors import generate_anchors
from utils.config import cfg
from utils.net_utils import bbox_overlaps, bbox_transform, torch_rand_choice


class AnchorTargetLayer(nn.Module):
    """
    Assign anchors to ground-truth targets. Produces anchor
    classification labels and bounding-box regression targets.
    """

    def __init__(self):
        super(AnchorTargetLayer, self).__init__()
        self.feat_stride = cfg.FEAT_STRIDE[0]
        self.anchors = generate_anchors()
        self.num_anchors = self.anchors.size(0)

    def forward(self, rpn_cls_score, gt_boxes, im_info):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        # filter out-of-image anchors
        # measure the overlaps between anchors and gt_boxes
        # assign label, bbox_targets, bbox_inside_weights, bbox_outside_weights for each anchor

        assert rpn_cls_score.size(0) == 1, 'Single batch only.'

        height, width = rpn_cls_score.shape[-2:]
        im_info = im_info[0]

        # Enumerate all shifts (NOTE: torch.meshgrid is different from np.meshgrid)
        shift_x = torch.arange(0, width) * self.feat_stride
        shift_y = torch.arange(0, height) * self.feat_stride
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        shift_x, shift_y = shift_x.contiguous(), shift_y.contiguous()
        shifts = torch.stack((shift_x.view(-1), shift_y.view(-1),
                              shift_x.view(-1), shift_y.view(-1)), dim=1)
        shifts = shifts.type_as(gt_boxes)

        # Enumerate all shifted anchors:
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get shift anchors (K, A, 4)
        # reshape to (K * A, 4) shifted anchors
        A = self.num_anchors
        K = shifts.size(0)
        self.anchors = self.anchors.type_as(gt_boxes)
        anchors = self.anchors.view(1, A, 4) + shifts.view(1, K, 4).permute(1, 0, 2)
        anchors = anchors.view(K * A, 4)

        # only keep anchors inside the image
        inds_inside = torch.nonzero((anchors[:, 0] >= 0) &
                                    (anchors[:, 1] >= 0) &
                                    (anchors[:, 2] < im_info[1]) &
                                    (anchors[:, 3] < im_info[0]))[:, 0]
        anchors = anchors[inds_inside]

        overlaps = bbox_overlaps(anchors, gt_boxes)

        if 'DEBUG' in os.environ:
            import numpy as np
            argmax_overlaps = np.argmax(overlaps.cpu(), axis=1)
            max_overlaps = torch.from_numpy(np.max(overlaps.cpu().numpy(), axis=1))
            gt_max_overlaps = torch.from_numpy(np.max(overlaps.cpu().numpy(), axis=0)).type_as(overlaps)
            gt_argmax_overlaps = np.nonzero(overlaps == gt_max_overlaps)[:, 0]
        else:
            argmax_overlaps = overlaps.argmax(dim=1)
            max_overlaps = overlaps.max(dim=1)[0]
            gt_max_overlaps = overlaps.max(dim=0)[0]
            gt_argmax_overlaps = torch.nonzero(overlaps == gt_max_overlaps)[:, 0]

        # label: 1 is positive, 0 is negative, -1 is dont care
        # The anchors which statisfied both positive and negative conditions will be as positive
        labels = gt_boxes.new(len(inds_inside)).fill_(-1)

        # bg labels: below threshold IOU
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        # subsample positive labels if we have too many
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        fg_inds = torch.nonzero(labels == 1)[:, 0]
        if len(fg_inds) > num_fg:
            if 'DEBUG' in os.environ:
                disable_inds = fg_inds[:len(fg_inds) - num_fg]
            else:
                disable_inds = torch_rand_choice(fg_inds, len(fg_inds) - num_fg)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = cfg.TRAIN.RPN_BATCHSIZE - torch.sum(labels == 1)
        bg_inds = torch.nonzero(labels == 0)[:, 0]
        if len(bg_inds) > num_bg:
            if 'DEBUG' in os.environ:
                disable_inds = bg_inds[:len(bg_inds) - num_bg]
            else:
                disable_inds = torch_rand_choice(bg_inds, len(bg_inds) - num_bg)
            labels[disable_inds] = -1

        # gt_boxes: (x1, y1, x2, y2, class, pid)
        bbox_targets = bbox_transform(anchors, gt_boxes[argmax_overlaps, :4])

        bbox_inside_weights = gt_boxes.new(bbox_targets.shape).zero_()
        bbox_inside_weights[labels == 1] = gt_boxes.new(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

        bbox_outside_weights = gt_boxes.new(bbox_targets.shape).zero_()
        num_examples = torch.sum(labels >= 0)
        bbox_outside_weights[labels == 1] = gt_boxes.new(1, 4).fill_(1) / num_examples
        bbox_outside_weights[labels == 0] = gt_boxes.new(1, 4).fill_(1) / num_examples

        def map2origin(data, count=K * A, inds=inds_inside, fill=0):
            """Map to original set."""
            shape = (count, ) + data.shape[1:]
            origin = torch.empty(shape).fill_(fill).type_as(gt_boxes)
            origin[inds] = data
            return origin

        labels = map2origin(labels, fill=-1)
        bbox_targets = map2origin(bbox_targets)
        bbox_inside_weights = map2origin(bbox_inside_weights)
        bbox_outside_weights = map2origin(bbox_outside_weights)

        labels = labels.view(1, height, width, A).permute(0, 3, 1, 2)
        labels = labels.contiguous().view(1, 1, A * height, width)
        bbox_targets = bbox_targets.view(1, height, width, A * 4).permute(0, 3, 1, 2)
        bbox_inside_weights = bbox_inside_weights.view(1, height, width, A * 4).permute(0, 3, 1, 2)
        bbox_outside_weights = bbox_outside_weights.view(1, height, width, A * 4).permute(0, 3, 1, 2)

        return labels, bbox_targets, bbox_inside_weights, bbox_outside_weights
