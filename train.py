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

from SSD_model import get_SSD_model
from VOC_data import VOC_dataset
from draw_img_utils import *
from SSDloss import *

torch.set_printoptions(precision=3)

PATH = 'C:\\datasets\\pascal\\'
anno_path = f'{PATH}PASCAL_VOC\\pascal_train2007.json'
train_dataset = VOC_dataset(PATH, anno_path)
batch_size = 16
learning_rate = 5e-4
vgg_weight_path = 'C:\\Users\\ruifr\\.torch\\models\\vgg16-397923af.pth'

def detection_collate_fn(batch):
    imgs, bboxes, labels = [], [], []
    for i, b, l in batch:
        imgs.append(i); bboxes.append(b); labels.append(l)
    return torch.stack(imgs), bboxes, labels

trn_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=detection_collate_fn)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

model = get_SSD_model(batch_size, vgg_weight_path)
model = model.to(device)

optimizer = torch.optim.SGD(params = model.parameters(), lr=1e-4, momentum=0.9)

for i, batch in enumerate(trn_dataloader):
    imgs, bboxes, labels = batch
    imgs = imgs.to(device)
    cls_preds, loc_preds = model(imgs)

    model.zero_grad()

    total_loss = 0
    total_loc_loss, total_cls_loss = 0, 0

    for _ in range(batch_size):
        
        img, bbox, label = imgs[_], bboxes[_], labels[_]
        cls_pred, loc_pred = cls_preds[_], loc_preds[_]

        # img, bbox, label = tmp
        # img, bbox, label = train_dataset[0]

        # img = img.unsqueeze(0)

        # cls_pred, loc_pred = model(img)
        # cls_pred, loc_pred = cls_pred.squeeze(0), loc_pred.squeeze(0)

        # PATH = 'C:\\datasets\\pascal\\'
        # anno_path = f'{PATH}PASCAL_VOC\\pascal_train2007.json'
        # train_dataset = VOC_dataset(PATH, anno_path)

        # img, bbox, label = train_dataset[7]
        # img = img.unsqueeze(0)

        prior_box = get_prior_box()
        iou = get_iou(bbox, prior_box)

        pos_mask, cls_target, bbox_target = get_target(iou, prior_box, img, bbox, label)
        pos_mask, cls_target, bbox_target = pos_mask.to(device), cls_target.to(device), bbox_target.to(device)

        # model = get_SSD_model(1)
        # cls_pred, loc_pred = model(img)
        # cls_pred, loc_pred = cls_pred.squeeze(0), loc_pred.squeeze(0)

        loss_loc, loss_cls = loss(cls_pred, loc_pred, pos_mask, cls_target, bbox_target)
        total_loc_loss += loss_loc; total_cls_loss += loss_cls

        total_loss += (loss_loc + loss_cls)

    total_loss /= float(batch_size)
    total_loss.backward()

    optimizer.step()
    if i % 5 == 0:
        print('cls_loss: {}, loc_loss: {}, loss: {}'.format(total_cls_loss / float(batch_size), total_loc_loss / float(batch_size), total_loss))

