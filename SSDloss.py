import os
import torch
from tqdm import tqdm_notebook
from matplotlib import pyplot as plt
from itertools import product
import json
from collections import defaultdict
import random

from torch import nn
from torch.autograd import Variable
from torch.functional import F
from torchvision import models
import torchvision
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2

# from fastai import transforms, model, dataset, conv_learner

from PIL import ImageDraw, ImageFont
from matplotlib import patches, patheffects
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

torch.set_printoptions(precision=3)

# from SSD_model import get_SSD_model
from VOC_data import VOC_dataset
from draw_img_utils import *
from Config import Config

def get_prior_box():

    feature_map = [38, 19, 10, 5, 3, 1]
    min_size = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88]
    max_size = min_size[1:] + [1.05]
    aspect_ratio = [[2], [2,3], [2,3], [2,3], [2], [2]]

    prior_box = []
    for k, f in enumerate(feature_map):
        for i, j in product(range(feature_map[k]), repeat=2):
            top, left = (i + 0.5) / f, (j + 0.5) / f
            h, w  = [], []
            h.append(min_size[k]); w.append(min_size[k])
            h.append((min_size[k] * max_size[k]) ** 0.5); w.append((min_size[k] * max_size[k]) ** 0.5)
            
            for ar in aspect_ratio[k]:
                h.append(min_size[k] * ar ** 0.5); w.append(min_size[k] / ar ** 0.5)
                h.append(min_size[k] / ar ** 0.5); w.append(min_size[k] * ar ** 0.5)
            
            for hi, wi in zip(h, w):
#                 the order of left and top is very important
                prior_box.append((left, top, hi, wi))
    prior_box = torch.tensor(prior_box)

    # make sure the prior_box in the range of [0, 1]
    prior_box = prior_box.clamp(0, 1)

    return prior_box


def get_iou(bbox, prior_box):
    '''
    Description:
    get iou of each image, shape: (number of objects, 8732)
    Adapated from https://github.com/amdegroot/ssd.pytorch
    Since every image contains different number of objects, the matrix for iou will be different
    We have to use a for loop to handle all the images in a batch

    Arguments:
    bbox: List[List[xyxy]], percentage
    prior_box: tensor, (8732, 4)
    '''
    gt = torch.FloatTensor(bbox.copy())
    prior_box_ = torch.cat(
        [prior_box[:, 0:2] - prior_box[:, 2:4] / 2,
        prior_box[:, 0:2] + prior_box[:, 2:4] / 2],
        dim=1
    )

    A = gt.size(0)
    B = prior_box_.size(0)
    
    max_xy = torch.min(gt[:, 2:].unsqueeze(1).expand(A, B, 2),
                       prior_box_[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(gt[:, :2].unsqueeze(1).expand(A, B, 2),
                       prior_box_[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    inter = inter[:, :, 0] * inter[:, :, 1]
    
    area_a = ((gt[:, 2]-gt[:, 0]) *
              (gt[:, 3]-gt[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((prior_box_[:, 2]-prior_box_[:, 0]) *
              (prior_box_[:, 3]-prior_box_[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter

    return inter / union


# #     gt = torch.FloatTensor(imgs_bbox[id])
#     gt = torch.FloatTensor(bbox.copy())
#     # gt = bbox.clone()
    
#     # rescale
#     gt /= 300
    
#     # transform gt
#     gt[:, 2:] = gt[:, :2] + gt[:, 2:]
    
#     # transform prior_box
# #     prior_box_ = torch.cat([prior_box[:, :2] - prior_box[:, 2:] / 2, prior_box[:, :2] + prior_box[:, 2:] / 2], dim=1)
#     prior_box_ = torch.cat([(prior_box[:, 1] - prior_box[:, 2] / 2).unsqueeze(1), 
#                             (prior_box[:, 0] - prior_box[:, 3] / 2).unsqueeze(1), 
#                             (prior_box[:, 1] + prior_box[:, 2] / 2).unsqueeze(1), 
#                             (prior_box[:, 0] + prior_box[:, 3] / 2).unsqueeze(1)], dim=1)
    
#     A = gt.size(0)
#     B = prior_box_.size(0)
    
#     max_xy = torch.min(gt[:, 2:].unsqueeze(1).expand(A, B, 2),
#                        prior_box_[:, 2:].unsqueeze(0).expand(A, B, 2))
#     min_xy = torch.max(gt[:, :2].unsqueeze(1).expand(A, B, 2),
#                        prior_box_[:, :2].unsqueeze(0).expand(A, B, 2))
#     inter = torch.clamp((max_xy - min_xy), min=0)
#     inter = inter[:, :, 0] * inter[:, :, 1]
    
#     area_a = ((gt[:, 2]-gt[:, 0]) *
#               (gt[:, 3]-gt[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
#     area_b = ((prior_box_[:, 2]-prior_box_[:, 0]) *
#               (prior_box_[:, 3]-prior_box_[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
#     union = area_a + area_b - inter
    
    return inter / union


def get_target(iou, prior_box, img, bbox, label):
    # 4. get the best matche of each priorbox

    best_each_prior_box = iou.argmax(0)
    cls_target = torch.tensor(label)[best_each_prior_box]
    bbox_target = torch.tensor(bbox)[best_each_prior_box]

    # get the positive tags of the image
    pos_mask = iou.max(0)[0] > 0.5

    # get the best match of each object
    best_each_object = iou.argmax(1)

    for i, k in enumerate(best_each_object):
        cls_target[k] = int(label[i])
        pos_mask[k] = 1
        bbox_target[k] = torch.FloatTensor(bbox[i])
        
    cls_target[1 - pos_mask] = 0

    # no need to add 1, cause already [0, 20] to 21 classes
    # cls_target[pos_mask] += 1
    bbox_target = encode(bbox_target, prior_box, [0.1, 0.2])

    return pos_mask, cls_target, bbox_target

def encode_bbox_deprecated(prior_box, bbox_target):
    '''
    deprecated method, us encode(pred_bbox, priors, variance) instead
    '''
    # 5. encode the bbox_target

    variance = [0.1, 0.2]

    bbox_target /= 300

    # get the gt center x and y
    cxcy = bbox_target[:, 0:2] + bbox_target[:, 2:4] / 2
    cxcy = (cxcy - reversed(prior_box[:, 0:2])) / (prior_box[:, 2:4] * variance[0])

    # get the gt weight and height
    wh = torch.log(bbox_target[:, 2:4] / prior_box[:, 2:4])
    wh /= variance[1]

    bbox_target = torch.cat((cxcy, wh), dim=1)

    return bbox_target

# test encode and decode module
def encode(bbox, priors, variance):
    '''
    Arguments:
    bbox: (8732, 4), size / 300, xyxy format
    priors:    (8732, 4), cxcy + wh format
    '''
    cxcy = (bbox[:, 2:4] + bbox[:, 0:2]) / 2
    cxcy = (cxcy - priors[:, 0:2]) / (priors[:, 2:4] * variance[0])
    
    wh = (bbox[:, 2:4] - bbox[:, 0:2])
    wh = torch.log(wh / priors[:, 2:4]) / variance[1]
    
    return torch.cat([cxcy, wh], dim=1)

def decode(bbox, priors, variance):
    cxcy =  bbox[:, 0:2] * variance[0] * priors[:, 2:4] + priors[:, 0:2]
    wh = torch.exp(bbox[:, 2:4] * variance[1]) * priors[:, 2:4]
    
    x1y1 = cxcy - wh / 2
    x2y2 = cxcy + wh / 2
    
    return torch.cat([x1y1, x2y2], 1)


def loss(cls_pred, loc_pred, pos_mask, cls_target, bbox_target):
    '''
    one image a time
    '''     

    # prior_box = get_prior_box()
    # iou = get_iou(bbox, prior_box)
    # pos_mask, cls_target, bbox_target = get_target(iou, prior_box, img, bbox, label)
    
    ratio = 3
    num_pos = torch.sum(pos_mask)
    num_neg = ratio * num_pos
    neg_mask = 1 - pos_mask

    if num_pos == 0:
        return torch.tensor(0)

    # loss of class including hard negative mining
    loc_criterion, cls_criterion = nn.SmoothL1Loss(reduce=False), nn.CrossEntropyLoss(reduce=False)

    loss_cls = cls_criterion(cls_pred, cls_target.long())
    
    loss_cls_pos = loss_cls[pos_mask]
    loss_cls_pos = torch.sum(loss_cls_pos)

    conf_loss_neg = loss_cls[neg_mask]
    _, neg_idx = torch.sort(conf_loss_neg, descending=True)
    loss_cls_neg = torch.sum(conf_loss_neg[neg_idx[:num_neg]])

    loss_cls = loss_cls_pos + loss_cls_neg

    # loss of location
    loss_loc = loc_criterion(loc_pred[pos_mask], bbox_target[pos_mask])
    loss_loc = torch.sum(loss_loc)

    return loss_loc / float(num_pos), loss_cls / float(num_pos)

def nms(conf_threshold, iou_threshold, top_k, conf, loc, priors):
    '''
    Description:
    greedy nms

    Arguments:
    conf_threshold: int, default=0.45
    iou_threshold: int, defualt=0.01
    top_k: int, default=5
    conf: 
    loc:
    priors
    '''
    # 1. get the conf_score, conf_cls, bboxes and areaes
    loc_ = decode(loc[0], priors, [0.1, 0.2]) * 300

    conf_ = F.softmax(conf[0])
    conf_score, conf_cls = torch.max(conf_[:, 1:], dim=1)
    conf_cls += 1

    conf_mask = conf_score > conf_threshold
    conf_score, conf_cls, loc_ = conf_score[conf_mask], conf_cls[conf_mask], loc_[conf_mask]

    conf_score, conf_idx = torch.sort(conf_score, descending=True)
    conf_cls, bboxes = conf_cls[conf_idx], loc_[conf_idx]

    wh = bboxes[:, 2:] - bboxes[:, :2]
    areaes = wh[:, 0] * wh[:, 1]

    # 2. get the result bbox and result class
    res_bbox, res_cls = [], []

    for idx in range(len(conf_score)):
        if conf_score[idx] != 0:
            res_bbox.append(bboxes[idx])
            res_cls.append(conf_cls[idx])
            
            for i_head in range(idx + 1, len(conf_score)):
                if conf_score[i_head] != 0:
                    max_xy = torch.max(bboxes[idx][:2], bboxes[i_head][:2])
                    min_xy = torch.min(bboxes[idx][2:], bboxes[i_head][2:])
                    wh = torch.clamp(min_xy - max_xy, min=0)
                    intersect = wh[0] * wh[1]
                    iou = intersect / (areaes[idx] + areaes[i_head] - intersect)
                    if iou > iou_threshold:
                        conf_score[i_head] = 0

    res_bbox, res_cls = res_bbox[:top_k], res_cls[:top_k]
    return res_bbox, res_cls

if __name__ == "__main__":
    config = Config('local')
    conf_threshold = 0.1
    iou_threshold = 0.45
    top_k = 5

    config = Config('local')
    ssd_model = get_SSD_model(1, config.vgg_weight_path, config.vgg_reduced_weight_path)
    ssd_model.load_trained_model(config.trained_path)
    test_dataset = VOC_dataset(config.voc2007_root, config.voc2012_root, config.voc2007_test_anno, 'test')

    idx = 19
    img, bbox, label = test_dataset[idx]

    conf, loc = ssd_model(img.unsqueeze(0))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    ssd_model = ssd_model.to(device)

    priors = get_prior_box()
    res_bbox, res_cls = nms(conf_threshold, iou_threshold, top_k, conf, loc)
    print(res_bbpx, res_cls)