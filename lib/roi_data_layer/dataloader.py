"""DataLoader for the network"""

import numpy as np

from faster_rcnn.config import cfg
from roi_data_layer.minibatch import get_minibatch

DEBUG = True


class DataLoader:
    """DataLoader for the network"""

    def __init__(self, roidb, num_classes=2):
        self._roidb = roidb
        self._num_classes = num_classes
        self._perm = None  # Index permutation
        self._cur = None  # Current pos
        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        if cfg.TRAIN.ASPECT_GROUPING:
            if DEBUG:
                self._perm = np.arange(len(self._roidb))
            else:
                widths = np.array([r["width"] for r in self._roidb])
                heights = np.array([r["height"] for r in self._roidb])
                horz = widths >= heights
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                vert_inds = np.where(vert)[0]
                inds = np.hstack((np.random.permutation(horz_inds), np.random.permutation(vert_inds)))
                inds = np.reshape(inds, (-1, 2))
                row_perm = np.random.permutation(np.arange(inds.shape[0]))
                inds = np.reshape(inds[row_perm, :], (-1,))
                self._perm = inds
        else:
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur : self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch."""
        db_inds = self.get_next_minibatch_inds()
        minibatch_db = [self._roidb[i] for i in db_inds]
        blobs = get_minibatch(minibatch_db, self._num_classes)

        for key in blobs:
            blobs[key] = blobs[key].astype(np.float32)

        return blobs
