"""
Author: https://github.com/ShuangLI59/person_search.git
Description: Evaluate network in psdb_test dataset.
"""

from datasets.factory import get_imdb
from models.network import Network
from test_gallery import detect_and_exfeat
from test_probe import exfeat
from utils import pickle


def main():
    imdb = get_imdb('psdb_test')

    # 1. Detect and extract features from all the gallery images in the imdb
    net = Network()
    gboxes, gfeatures = detect_and_exfeat(net, imdb)

    # 2. Only extract features from given probe rois
    pfeatures = exfeat(net, imdb.probes)

    # Save
    pickle(gboxes, 'gallery_detections.pkl')
    pickle(gfeatures, 'gallery_features.pkl')
    pickle(pfeatures, 'probe_features.pkl')

    # Evaluate
    imdb.evaluate_detections(gboxes, det_thresh=0.5)
    imdb.evaluate_detections(gboxes, det_thresh=0.5, labeled_only=True)
    imdb.evaluate_search(gboxes, gfeatures['feat'], pfeatures['feat'], det_thresh=0.5,
                         gallery_size=100, dump_json='results.json')


if __name__ == "__main__":
    main()
