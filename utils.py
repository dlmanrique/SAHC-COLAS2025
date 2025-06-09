import numpy as np
import random
import torch
import os
import torch.nn.functional as F
import torch.nn as nn

# Configure device and seed everithing for reproducibility
seed = 19980125

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"



def label2phase(labels, phase2label_dict):
    label2phase_dict = {phase2label_dict[k]:k for k in phase2label_dict.keys()}
    phases = [label2phase_dict[label] for label in labels]
    return phases

max_pool = nn.MaxPool1d(kernel_size=13,stride=5,dilation=3)


def fusion(predicted_list,labels,args):
    # Part d of the general method diagram
    all_out_list = []
    resize_out_list = []
    labels_list = []
    all_out = 0
    len_layer = len(predicted_list)
    weight_list = [1.0/len_layer for i in range (0, len_layer)]

    
    # Iterate through the predicted outputs. This is F^{-}, F1, F2 and F3
    for out, w in zip(predicted_list, weight_list):

        resize_out = F.interpolate(out, size=labels.size(0), mode='nearest')
        resize_out_list.append(resize_out)
        resize_label = F.interpolate(labels.float().unsqueeze(0).unsqueeze(0), size=out.size(2), mode='linear', align_corners=False)
        if out.size(2)==labels.size(0):
            resize_label = labels
            labels_list.append(resize_label.squeeze().long())
        else:
            
            resize_label = F.interpolate(labels.float().unsqueeze(0).unsqueeze(0), size=out.size(2), mode='nearest')
            labels_list.append(resize_label.squeeze().long())

        all_out_list.append(out)

    return all_out, all_out_list, labels_list


def cosine_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim==1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim==2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    similiarity = np.dot(a, b.T)/(a_norm * b_norm)
    # dist = 1. - similiarity
    return similiarity



