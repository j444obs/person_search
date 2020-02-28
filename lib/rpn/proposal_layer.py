# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os

import torch
import torch.nn as nn
from torchvision.ops import nms

from rpn.generate_anchors import generate_anchors
from utils.config import cfg
from utils.net_utils import bbox_transform_inv, clip_boxes, filter_boxes


class ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self):
        super(ProposalLayer, self).__init__()
        self.feat_stride = cfg.FEAT_STRIDE[0]
        self.anchors = generate_anchors()
        self.num_anchors = self.anchors.size(0)

    def forward(self, scores, bbox_deltas, im_info):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals

        assert scores.size(0) == 1, "Single batch only."

        cfg_key = "TRAIN" if self.training else "TEST"
        pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
        min_size = cfg[cfg_key].RPN_MIN_SIZE

        # the first set of num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = scores[:, self.num_anchors :, :, :]
        im_info = im_info[0]

        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.shape[-2:]

        # Enumerate all shifts (NOTE: torch.meshgrid is different from np.meshgrid)
        shift_x = torch.arange(0, width) * self.feat_stride
        shift_y = torch.arange(0, height) * self.feat_stride
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        shift_x, shift_y = shift_x.contiguous(), shift_y.contiguous()
        shifts = torch.stack(
            (shift_x.view(-1), shift_y.view(-1), shift_x.view(-1), shift_y.view(-1)), dim=1
        )
        shifts = shifts.type_as(scores)

        # Enumerate all shifted anchors:
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get shift anchors (K, A, 4)
        # reshape to (K * A, 4) shifted anchors
        A = self.num_anchors
        K = shifts.size(0)
        self.anchors = self.anchors.type_as(scores)
        anchors = self.anchors.view(1, A, 4) + shifts.view(1, K, 4).permute(1, 0, 2)
        anchors = anchors.view(K * A, 4)

        # Permute and reshape predicted bbox transformations to the same order as the anchors:
        # bbox deltas will be (1, 4 * A, H, W) format
        # permute to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4)
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous().view(-1, 4)

        # Safe-guard for unexpected large dw or dh value.
        # Since our proposals are only human, some background region features will never
        # receive gradients from bbox regression. Thus their predictions may drift away.
        bbox_deltas[:, 2:].clamp_(-10, 10)

        # Same story for the scores:
        # scores are (1, A, H, W) format
        # permute to (1, H, W, A)
        # reshape to (1 * H * W * A, 1)
        scores = scores.permute(0, 2, 3, 1).contiguous().view(-1, 1)

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)

        # 2. Clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info)

        # 3. Remove predicted boxes with either height or width < threshold
        # (NOTE: need to scale min_size with the input image scale stored in im_info[2])
        keep = filter_boxes(proposals, min_size * im_info[2])
        proposals = proposals[keep]
        scores = scores[keep]

        # 4. Sort all (proposal, score) pairs by score from highest to lowest
        # 5. Take top pre_nms_topN (e.g. 6000)
        if "DEBUG" in os.environ:
            import numpy as np

            order = np.argsort(scores.view(-1).cpu()).numpy()[::-1]
            order = torch.from_numpy(order.copy())
        else:
            order = scores.view(-1).argsort(descending=True)
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order]
        scores = scores[order]

        # 6. Apply nms (e.g. threshold = 0.7)
        # 7. Take after_nms_topN (e.g. 300)
        # 8. Return the top proposals
        if "DEBUG" in os.environ:
            keep = torch.arange(proposals.shape[0])
        else:
            keep = nms(proposals, scores.squeeze(1), nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep]
        scores = scores[keep]

        # Output proposals in [img_id, x1, y1, x2, y2] format.
        # Our RPN implementation only supports a single input image, so all img_ids are 0.
        output = torch.zeros(proposals.size(0), 5).type_as(scores)
        output[:, 1:] = proposals
        return output
