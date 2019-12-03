"""
@Author: Ross Girshick
@Description: Tools about bounding box
"""

import numpy as np
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
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack((targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets


def bbox_transform_inv(boxes, deltas):
    """Apply transformer on the boxes."""
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1  # x2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1  # y2

    return pred_boxes


def clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)  # x1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)  # y1 >= 0
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)  # x2 < im_shape[1]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)  # y2 < im_shape[0]
    return boxes


def bbox_overlaps(anchors, gt_boxes):
    """Compute the overlaps between anchors and gt_boxes."""
    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_area = ((gt_boxes[:, 2] - gt_boxes[:, 0] + 1) *
                     (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)).view(1, K)

    anchors_area = ((anchors[:, 2] - anchors[:, 0] + 1) *
                    (anchors[:, 3] - anchors[:, 1] + 1)).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (torch.min(boxes[:, :, 2], query_boxes[:, :, 2]) -
          torch.max(boxes[:, :, 0], query_boxes[:, :, 0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:, :, 3], query_boxes[:, :, 3]) -
          torch.max(boxes[:, :, 1], query_boxes[:, :, 1]) + 1)
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps
