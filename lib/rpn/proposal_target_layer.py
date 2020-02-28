# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os

import torch
import torch.nn as nn

from utils.config import cfg
from utils.net_utils import bbox_overlaps, bbox_transform, torch_rand_choice


class ProposalTargetLayer(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces
    proposal classification labels and bounding-box regression targets.
    """

    def __init__(self, num_classes, bg_pid_label=5532):
        super(ProposalTargetLayer, self).__init__()
        self.num_classes = num_classes
        self.bg_pid_label = bg_pid_label

    def forward(self, all_rois, gt_boxes):
        # all_rois: region proposals in (0, x1, y1, x2, y2) format coming from RPN
        # gt_boxes: (x1, y1, x2, y2, class, pid)

        # Include ground-truth boxes in the set of candidate rois
        zeros = gt_boxes.new(gt_boxes.shape[0], 1).zero_()
        all_rois = torch.cat((all_rois, torch.cat((zeros, gt_boxes[:, :4]), dim=1)), dim=0)

        # Sanity check: single batch only
        assert torch.all(all_rois[:, 0] == 0), "Single batch only."

        num_rois = cfg.TRAIN.BATCH_SIZE
        num_fg_rois = round(cfg.TRAIN.FG_FRACTION * num_rois)

        overlaps = bbox_overlaps(all_rois[:, 1:5], gt_boxes[:, :4])
        argmax_overlaps = overlaps.argmax(dim=1)
        max_overlaps = overlaps.max(dim=1)[0]
        labels = gt_boxes[argmax_overlaps, 4]
        pid_labels = gt_boxes[argmax_overlaps, 5]

        # Sample foreground RoIs
        fg_inds = torch.nonzero(max_overlaps >= cfg.TRAIN.FG_THRESH)[:, 0]
        num_fg_rois = min(num_fg_rois, fg_inds.numel())
        if fg_inds.numel() > 0:
            if "DEBUG" in os.environ:
                fg_inds = fg_inds[:num_fg_rois]
            else:
                fg_inds = torch_rand_choice(fg_inds, num_fg_rois)

        # Sample background RoIs
        bg_inds = torch.nonzero(
            (max_overlaps < cfg.TRAIN.BG_THRESH_HI) & (max_overlaps >= cfg.TRAIN.BG_THRESH_LO)
        )[:, 0]
        num_bg_rois = min(num_rois - num_fg_rois, bg_inds.numel())
        if bg_inds.numel() > 0:
            if "DEBUG" in os.environ:
                bg_inds = bg_inds[:num_bg_rois]
            else:
                bg_inds = torch_rand_choice(bg_inds, num_bg_rois)

        assert num_fg_rois + num_bg_rois == num_rois

        keep_inds = torch.cat((fg_inds, bg_inds))
        labels = labels[keep_inds]
        pid_labels = pid_labels[keep_inds]
        rois = all_rois[keep_inds]

        # Correct the labels and pid_labels of bg rois
        labels[num_fg_rois:] = 0
        pid_labels[num_fg_rois:] = self.bg_pid_label

        bbox_targets_data = bbox_transform(rois[:, 1:5], gt_boxes[argmax_overlaps][keep_inds, :4])
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            means = gt_boxes.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
            stds = gt_boxes.new(cfg.TRAIN.BBOX_NORMALIZE_STDS)
            bbox_targets_data = (bbox_targets_data - means) / stds

        regression_targets = self.get_regression_targets(
            bbox_targets_data, labels, self.num_classes
        )
        bbox_targets, bbox_inside_ws, bbox_outside_ws = regression_targets

        return rois, labels.long(), pid_labels.long(), bbox_targets, bbox_inside_ws, bbox_outside_ws

    @staticmethod
    def get_regression_targets(bbox_target_data, labels, num_classes):
        """
        Given targets in [N, 4] format, get bbox regression targets
        in [N, 4 * k] format (only one class has non-zero targets).
        """
        bbox_targets = labels.new(labels.numel(), 4 * num_classes).zero_()
        bbox_inside_weights = labels.new(bbox_targets.shape).zero_()
        bbox_outside_weights = labels.new(bbox_targets.shape).zero_()
        fg_inds = torch.nonzero(labels > 0)[:, 0]
        for ind in fg_inds:
            cls = int(torch.round(labels[ind]))
            start = 4 * cls
            end = start + 4
            bbox_targets[ind, start:end] = bbox_target_data[ind]
            bbox_inside_weights[ind, start:end] = labels.new(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)
            bbox_outside_weights[ind, start:end] = labels.new(4).fill_(1)
        return bbox_targets, bbox_inside_weights, bbox_outside_weights
