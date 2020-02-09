"""
Author: Ross Girshick
Description: Tools for training network.
"""

import torch


def bbox_transform(ex_rois, gt_rois):
    """Compute bounding box transform target."""
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = torch.log(gt_widths / ex_widths)
    targets_dh = torch.log(gt_heights / ex_heights)

    targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets


def bbox_transform_inv(boxes, deltas):
    """Apply transformer on the boxes."""
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
    pred_w = torch.exp(dw) * widths.unsqueeze(1)
    pred_h = torch.exp(dh) * heights.unsqueeze(1)

    pred_boxes = deltas.clone()
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1  # x2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1  # y2

    return pred_boxes


def clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    boxes[:, 0::4].clamp_(0, im_shape[1] - 1)
    boxes[:, 1::4].clamp_(0, im_shape[0] - 1)
    boxes[:, 2::4].clamp_(0, im_shape[1] - 1)
    boxes[:, 3::4].clamp_(0, im_shape[0] - 1)
    return boxes


def bbox_overlaps(boxes, query_boxes):
    """Compute the overlaps between anchors and gt_boxes."""
    box_areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    query_areas = ((query_boxes[:, 2] - query_boxes[:, 0] + 1) *
                   (query_boxes[:, 3] - query_boxes[:, 1] + 1))

    iw = (torch.min(boxes[:, 2:3], query_boxes[:, 2:3].t()) -
          torch.max(boxes[:, 0:1], query_boxes[:, 0:1].t()) + 1).clamp(min=0)
    ih = (torch.min(boxes[:, 3:4], query_boxes[:, 3:4].t()) -
          torch.max(boxes[:, 1:2], query_boxes[:, 1:2].t()) + 1).clamp(min=0)
    ua = box_areas.view(-1, 1) + query_areas.view(1, -1) - iw * ih
    overlaps = iw * ih / ua

    return overlaps


def filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = torch.nonzero((ws >= min_size) & (hs >= min_size))[:, 0]
    return keep


# def smooth_l1_loss(pred, targets, inside_ws, outside_ws, sigma=1):
#     """
#     Compute smooth l1 loss for faster-rcnn network.

#         f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
#                |x| - 0.5 / sigma / sigma    otherwise
#     """
#     sigma_2 = sigma ** 2
#     x = inside_ws * (pred - targets)
#     sign = (x.abs() < 1 / sigma_2).detach().float()
#     loss = 0.5 * sigma_2 * x.pow(2) * sign + (x.abs() - 0.5 / sigma_2) * (1 - sign)
#     loss = outside_ws * loss
#     return loss.sum()

def smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=None):
    if dim is None:
        dim = [1]
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
        + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
        loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box


def torch_rand_choice(arr, size):
    """Generates a random sample from a given array, like numpy.random.choice."""
    idxs = torch.randperm(arr.size(0))[:size]
    return arr[idxs]
