import pickle

import torch

from models.network import Network


def init_from_caffe(net):
    dict_new = net.state_dict().copy()
    weight_path = "/home/zjli/Desktop/person_search/pkls/resnet50_caffe.pkl"
    caffe_weights = pickle.load(open(weight_path, "rb"), encoding="latin1")
    for k in net.state_dict():
        splits = k.split(".")

        if splits[-2] in ["labeled_matching_layer", "unlabeled_matching_layer"]:
            continue

        # Layer name mapping
        if splits[-2] == "rpn_conv":
            name = "rpn_conv/3x3"
        elif splits[-2] == "cls_score":
            name = "det_score"
        elif splits[-2] in ["rpn_cls_score", "rpn_bbox_pred", "bbox_pred", "feat_lowdim"]:
            name = splits[-2]
        else:
            name = "caffe." + splits[-2]

        if name not in caffe_weights:
            print("Layer: %s not found" % k)
            continue

        if splits[-1] == "weight":  # For BN, weight is scale
            dict_new[k] = torch.from_numpy(caffe_weights[name][0]).reshape(dict_new[k].shape)
        elif splits[-1] == "bias":  # For BN, bias is shift
            dict_new[k] = torch.from_numpy(caffe_weights[name][1]).reshape(dict_new[k].shape)
        elif splits[-1] == "running_mean":
            dict_new[k] = torch.from_numpy(caffe_weights[name][2]).reshape(dict_new[k].shape)
        elif splits[-1] == "running_var":
            dict_new[k] = torch.from_numpy(caffe_weights[name][3]).reshape(dict_new[k].shape)
        elif splits[-1] == "num_batches_tracked":  # num_batches_tracked is unuseful in test phase
            continue
        else:
            print("Layer: %s not found" % k)
            continue

    net.load_state_dict(dict_new)
    print("Load caffe model successfully!")


net = Network()
init_from_caffe(net)
torch.save(net.state_dict(), "resnet50_caffe.pth")
