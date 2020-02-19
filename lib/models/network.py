import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign, RoIPool

from models.base_feat_layer import BaseFeatLayer
from models.proposal_feat_layer import ProposalFeatLayer
from oim.labeled_matching_layer import LabeledMatchingLayer
from oim.unlabeled_matching_layer import UnlabeledMatchingLayer
from rpn.proposal_target_layer import ProposalTargetLayer
from rpn.rpn_layer import RPN
from utils.config import cfg
from utils.net_utils import smooth_l1_loss


class Network(nn.Module):
    """Person search network."""

    def __init__(self, pretrained_model=None):
        super(Network, self).__init__()
        rpn_depth = 1024  # Depth of the feature map fed into RPN
        num_classes = 2   # Background and foreground

        # Extracting feature layer
        self.base_feat_layer = BaseFeatLayer()
        self.proposal_feat_layer = ProposalFeatLayer()

        # RPN
        self.rpn = RPN(rpn_depth)
        self.proposal_target_layer = ProposalTargetLayer(num_classes=num_classes)
        self.rois = None  # proposals produced by RPN

        # Pooling layer
        pool_size = cfg.POOLING_SIZE
        self.roi_align = RoIAlign((pool_size, pool_size), 1.0 / 16.0, 0)
        self.roi_pool = RoIPool((pool_size, pool_size), 1.0 / 16.0)

        # Identification layer
        self.cls_score = nn.Linear(2048, num_classes)
        self.bbox_pred = nn.Linear(2048, num_classes * 4)
        self.feat_lowdim = nn.Linear(2048, 256)
        self.labeled_matching_layer = LabeledMatchingLayer()
        self.unlabeled_matching_layer = UnlabeledMatchingLayer()

        if pretrained_model:
            state_dict = torch.load(pretrained_model)
            self.load_state_dict({k: v for k, v in state_dict.items() if k in self.state_dict()})
            print("Loaded pretrained model from: %s" % pretrained_model)

        self.frozen_blocks()

    def forward(self, im_data, im_info, gt_boxes, is_prob=False, rois=None):
        assert im_data.size(0) == 1, 'Single batch only.'

        # Extract basic feature from image data
        base_feat = self.base_feat_layer(im_data)

        if not is_prob:
            # Feed base feature map to RPN to obtain rois
            self.rois, rpn_loss_cls, rpn_loss_bbox = self.rpn(base_feat, im_info, gt_boxes)
        else:
            assert rois is not None, "RoIs is not given in detect probe mode."
            self.rois, rpn_loss_cls, rpn_loss_bbox = rois, 0, 0

        if self.training:
            # Sample 128 rois and assign them labels and bbox regression targets
            roi_data = self.proposal_target_layer(self.rois, gt_boxes)
            self.rois, rois_label, pid_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
        else:
            rois_label, pid_label, rois_target, rois_inside_ws, rois_outside_ws = [None] * 5

        # Do roi pooling based on region proposals
        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.roi_align(base_feat, self.rois)
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.roi_pool(base_feat, self.rois)
        else:
            raise NotImplementedError("Only support roi_align and roi_pool.")

        # Extract the features of proposals
        proposal_feat = self.proposal_feat_layer(pooled_feat).squeeze()
        if is_prob:
            proposal_feat = proposal_feat.unsqueeze(0)

        cls_score = self.cls_score(proposal_feat)
        cls_prob = F.softmax(cls_score, dim=1)
        bbox_pred = self.bbox_pred(proposal_feat)
        feat_lowdim = self.feat_lowdim(proposal_feat)
        feat = F.normalize(feat_lowdim)

        if self.training:
            loss_cls = F.cross_entropy(cls_score, rois_label)
            loss_bbox = smooth_l1_loss(bbox_pred,
                                       rois_target,
                                       rois_inside_ws,
                                       rois_outside_ws)

            # OIM loss
            labeled_matching_scores, id_labels = self.labeled_matching_layer(feat, pid_label)
            labeled_matching_scores *= 10
            unlabeled_matching_scores = self.unlabeled_matching_layer(feat, pid_label)
            unlabeled_matching_scores *= 10
            id_scores = torch.cat((labeled_matching_scores, unlabeled_matching_scores), dim=1)
            loss_id = F.cross_entropy(id_scores, id_labels, ignore_index=-1)
        else:
            loss_cls, loss_bbox, loss_id = 0, 0, 0

        return cls_prob, bbox_pred, feat, rpn_loss_cls, rpn_loss_bbox, loss_cls, loss_bbox, loss_id

    def frozen_blocks(self):
        for p in self.base_feat_layer.SpatialConvolution_0.parameters():
            p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False

        # Frozen all bn layers in base_feat_layer
        self.base_feat_layer.apply(set_bn_fix)

    def train(self, mode=True):
        nn.Module.train(self, mode)

        if mode:
            # Set all bn layers in base_feat_layer to eval mode
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.base_feat_layer.apply(set_bn_eval)
