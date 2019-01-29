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

def get_prior_box():

    feature_map = [38, 19, 10, 5, 3, 1]
    min_size = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88]
    max_size = min_size[1:] + [1.05]
    aspect_ratio = [[2], [2,3], [2,3], [2,3], [2], [2]]

    prior_box = []
    for k, f in enumerate(feature_map):
        for i, j in product(range(feature_map[k]), repeat=2):
            cx, cy = (i + 0.5) / f, (j + 0.5) / f
            h, w  = [], []
            h.append(min_size[k]); w.append(min_size[k])
            h.append((min_size[k] * max_size[k]) ** 0.5); w.append((min_size[k] * max_size[k]) ** 0.5)
            
            for ar in aspect_ratio[k]:
                h.append((min_size[k] / ar) ** 0.5); w.append((min_size[k] * ar) ** 0.5)
                h.append((min_size[k] * ar) ** 0.5); w.append((min_size[k] / ar) ** 0.5)
            
            for hi, wi in zip(h, w):
                prior_box.append((cx, cy, hi, wi))
    prior_box = torch.tensor(prior_box)

    return prior_box


def get_iou(bbox, prior_box):
    '''
    Description:
    get iou of each image, shape: (number of objects, 8732)

    Adapated from https://github.com/amdegroot/ssd.pytorch

    Since every image contains different number of objects, the matrix for iou will be different
    We have to use a for loop to handle all the images in a batch


    imgs_bbox: List[List[cx, cy, w, h]]
    prior_box: tensor, (8732, 4)
    '''
#     gt = torch.FloatTensor(imgs_bbox[id])
    gt = torch.FloatTensor(bbox.copy())
    # gt = bbox.clone()
    
    # rescale
    gt /= 300
    
    # transform gt
    gt[:, 2:] = gt[:, :2] + gt[:, 2:]
    
    # transform prior_box
#     prior_box_ = torch.cat([prior_box[:, :2] - prior_box[:, 2:] / 2, prior_box[:, :2] + prior_box[:, 2:] / 2], dim=1)
    prior_box_ = torch.cat([(prior_box[:, 1] - prior_box[:, 2] / 2).unsqueeze(1), 
                            (prior_box[:, 0] - prior_box[:, 3] / 2).unsqueeze(1), 
                            (prior_box[:, 1] + prior_box[:, 2] / 2).unsqueeze(1), 
                            (prior_box[:, 0] + prior_box[:, 3] / 2).unsqueeze(1)], dim=1)
    
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
    bbox_target = encode_bbox(prior_box, bbox_target)

    return pos_mask, cls_target, bbox_target


def encode_bbox(prior_box, bbox_target):
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

    if num_pos == 0:
        return torch.tensor(0)

    # loss of class including hard negative mining
    loc_criterion, cls_criterion = nn.SmoothL1Loss(), nn.CrossEntropyLoss(reduce=False)
    
    loss_cls_pos = cls_criterion(cls_pred[pos_mask], cls_target.long()[pos_mask])
    loss_cls_pos = torch.sum(loss_cls_pos)

    conf_loss_neg = cls_criterion(cls_pred[1 - pos_mask], cls_target.long()[1 - pos_mask])
    val, arg = torch.sort(conf_loss_neg, descending=True)
    loss_cls_neg = torch.sum(conf_loss_neg[arg[:num_neg]])

#     loss_cls = loss_cls_neg + loss_cls_pos 

#     without hard negative mining
    loss_cls = loss_cls_pos

    # loss of location
    loss_loc = loc_criterion(loc_pred[pos_mask], bbox_target[pos_mask])
    # loss = loss_loc + loss_cls / float(num_pos)
    
    # print(loss_cls.data / float(num_pos), loss_loc)
    
    # print(cls_pred[pos_mask][0], cls_target[pos_mask][0])
    # print(cls_criterion(cls_pred[pos_mask][0], cls_target[pos_mask][0]))

    return loss_loc, loss_cls / float(num_pos)


if __name__ == "__main__":
    PATH = 'C:\\datasets\\pascal\\'
    anno_path = f'{PATH}PASCAL_VOC\\pascal_train2007.json'
    train_dataset = VOC_dataset(PATH, anno_path)

    img, bbox, label = train_dataset[7]
    img = img.unsqueeze(0)

    prior_box = get_prior_box()
    iou = get_iou(bbox, prior_box)

    pos_mask, cls_target, bbox_target = get_target(iou, prior_box, img, bbox, label)
    
    model = get_SSD_model(1)
    cls_pred, loc_pred = model(img)
    cls_pred, loc_pred = cls_pred.squeeze(0), loc_pred.squeeze(0)

    loss(cls_pred, loc_pred, pos_mask, cls_target, bbox_target)