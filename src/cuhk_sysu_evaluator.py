import logging

import numpy as np
import torch
from detectron2.evaluation import DatasetEvaluator
from sklearn.metrics import average_precision_score

# NOTE: Only support one GPU


class CUHK_SYSU_Evaluator(DatasetEvaluator):
    def __init__(self, dataset):
        self.dataset = dataset

    def reset(self):
        self.predictions = []

    def process(self, inputs, outputs):
        for _, output in zip(inputs, outputs):
            instances = output["instances"]
            boxes = instances.pred_boxes.tensor
            scores = instances.scores
            prediction = torch.cat((boxes, scores.unsqueeze(1)), dim=1).cpu().numpy()
            self.predictions.append(prediction)

    def evaluate(self):
        evaluate_detections(self.dataset, self.predictions)
        return "Nothing to return"


def compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union


def evaluate_detections(dataset, gallery_det, threshold=0.5, iou_thresh=0.5, labeled_only=False):
    """
    gallery_det (list of ndarray): n_det x [x1, x2, y1, y2, score] per image
    threshold (float): filter out gallery detections whose scores below this
    iou_thresh (float): treat as true positive if IoU is above this threshold
    labeled_only (bool): filter out unlabeled background people
    """
    assert dataset.num_images == len(gallery_det)

    roidb = dataset.roidb
    y_true, y_score = [], []
    count_gt, count_tp = 0, 0
    for gt, det in zip(roidb, gallery_det):
        gt_boxes = gt["gt_boxes"]
        if labeled_only:
            inds = np.where(gt["gt_pids"].ravel() > 0)[0]
            if len(inds) == 0:
                continue
            gt_boxes = gt_boxes[inds]
        det = np.asarray(det)
        inds = np.where(det[:, 4].ravel() >= threshold)[0]
        det = det[inds]
        num_gt = gt_boxes.shape[0]
        num_det = det.shape[0]
        if num_det == 0:
            count_gt += num_gt
            continue
        ious = np.zeros((num_gt, num_det), dtype=np.float32)
        for i in range(num_gt):
            for j in range(num_det):
                ious[i, j] = compute_iou(gt_boxes[i], det[j, :4])
        tfmat = ious >= iou_thresh
        # for each det, keep only the largest iou of all the gt
        for j in range(num_det):
            largest_ind = np.argmax(ious[:, j])
            for i in range(num_gt):
                if i != largest_ind:
                    tfmat[i, j] = False
        # for each gt, keep only the largest iou of all the det
        for i in range(num_gt):
            largest_ind = np.argmax(ious[i, :])
            for j in range(num_det):
                if j != largest_ind:
                    tfmat[i, j] = False
        for j in range(num_det):
            y_score.append(det[j, -1])
            y_true.append(tfmat[:, j].any())
        count_tp += tfmat.sum()
        count_gt += num_gt

    det_rate = count_tp * 1.0 / count_gt
    ap = average_precision_score(y_true, y_score) * det_rate

    logging.info("{} detection:".format("Labeled only" if labeled_only else "All"))
    logging.info("  Recall = {:.2%}".format(det_rate))
    if not labeled_only:
        logging.info("  AP = {:.2%}".format(ap))
