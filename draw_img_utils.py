import os
import torch
from tqdm import tqdm_notebook
from matplotlib import pyplot as plt
from itertools import product
import json
from collections import defaultdict

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

# PATH = 'C:\\datasets\\pascal\\'
# trn_json = json.load(open(f'{PATH}PASCAL_VOC\\pascal_train2007.json')) 
# val_json = json.load(open(f'{PATH}PASCAL_VOC\\pascal_val2007.json')) 

PATH = '/home/kindeqi/PyTorch_SSD/dataset/VOCdevkit/VOC2007'
trn_anno_path = '/home/kindeqi/PyTorch_SSD/annotation/PASCAL_VOC/pascal_train2007.json'
val_anno_path = '/home/kindeqi/PyTorch_SSD/annotation/PASCAL_VOC/pascal_val2007.json'
trn_json = json.load(open(trn_anno_path))
val_json = json.load(open(val_anno_path))

def show_img(im, figsize=None, ax=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
    return ax

def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])
    
def draw_rect(ax, b, format='xywh'):
    if format=='xywh':
        patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor='white', lw=2))
    if format=='xyxy':
        patch = ax.add_patch(patches.Rectangle(b[:2], *(b[2:4] - b[0:2]), fill=False, edgecolor='white', lw=2)) 
#     patch = ax.add_patch(patches.Rectangle([0, 0], 0, 100, fill=False, edgecolor='white', lw=2))
    draw_outline(patch, 4)
    
def draw_text(ax, xy, txt, sz=14):
    text = ax.text(*xy, txt,
        verticalalignment='top', color='white', fontsize=sz, weight='bold')
    draw_outline(text, 1)
    
def draw_im(im, ann, cats):
    ax = show_img(im, figsize=(16,8))
    for b,c in ann:
#         b = bb_hw(b)
        draw_rect(ax, b)
        draw_text(ax, b[:2], cats[c], sz=16)

def draw_im_with_data(im, bbox, label, idx_category, bbox_format='xyxy'):
    '''
    Arguments:
    im: image, np.float32, (w, h, c)
    bbox: bounding box, np.float32, (k, 4)
    label: label, np.int32, (k, 1)
    
    Return:
    None
    '''
    ax = show_img(im, figsize=(16, 8))
    for b, c in zip(bbox, label):
        draw_rect(ax, b, bbox_format)
        draw_text(ax, b[:2], idx_category[c], sz=16)
        
def draw_idx(i, trn_id_fname, trn_id_annotation, idx_category):
    '''
    Arguments:
    trn_id_fname: dict{id: fname}
    i: int, img_id
    trn_anno: List[List[List, int]]
    idx_category: dict{id: category}
    
    Return:
    None
    '''
    im_a = trn_id_annotation[i]
    im = plt.imread(f'{PATH}\\JPEGIMAGES\\{trn_id_fname[i]}')
    print(im.shape)
    draw_im(im, im_a, idx_category)
    
def bbox_str(bbox):
    '''
    Arguments:
    bbox: List(int)
    
    Return:
    str
    '''
    
    return ' '.join(str(ele) for ele in bbox)

def show_multi_box(imgs):
    '''
    imgs: torch.tensor, b,c,h,w
    
    '''
    imgs_ = imgs.permute(0, 2, 3, 1)
    imgs_ = np.array(imgs_)
    row, col = int(len(imgs) ** 0.5) + 1, int(len(imgs) ** 0.5) + 1
    fig=plt.figure(figsize=(50, 50))
    print(row, col)
    for i in range(1, len(imgs) +1):
        img = imgs_[i - 1]
#         print(img.shape)
        fig.add_subplot(row, col, i)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

def get_anno_data():
    trn_id_fname = {img['id']: img['file_name'] for img in trn_json['images']}

    trn_id_annotation = defaultdict(list)
    for anno in trn_json['annotations']:
        trn_id_annotation[anno['image_id']].append([anno['bbox'], anno['category_id']])

    trn_id_single_anno = defaultdict(list)
    for anno in trn_id_annotation:
        for bbox, c in trn_id_annotation[anno]:
            if trn_id_single_anno[anno] == [] or bbox[-1] * bbox[-2] > trn_id_single_anno[anno][0][0][-1] * trn_id_single_anno[anno][0][0][-2]:
                trn_id_single_anno[anno] = [[bbox, c]]
                
    idx_category = {tmp['id']: tmp['name'] for tmp in trn_json['categories']}
    category_idx = {tmp['name']: tmp['id'] for tmp in trn_json['categories']}

    # ---------------------------------------------------------------

    batch_size = 8
    start_idx = 8
    fnames = [trn_id_fname[k] for k in trn_id_fname.keys()][start_idx:8 + start_idx]
    imgs = [cv2.imread(f'{PATH}JPEGImages\\{fname}') for fname in fnames]

    imgs_ratio = [[300 / img.shape[1], 300 / img.shape[0]] for img in imgs]
    imgs_size = [img.shape[:2] for img in imgs]

    imgs = [cv2.resize(img, (300, 300)) for img in imgs]
    imgs = torch.tensor(imgs)
    imgs = imgs.permute(0, 3, 1, 2)
    imgs = imgs.float() / 255

    imgs_id = [k for k in trn_id_fname.keys()][start_idx:8 + start_idx]
    imgs_bbox = [[ann[0] for ann in trn_id_annotation[i]] for i in imgs_id]

    for idx, img_ in enumerate(imgs_bbox):
        tmp = []
        for bbox in img_:
            y, x, width, height = bbox
            y, width = y * imgs_ratio[idx][0], width * imgs_ratio[idx][0]
            x, height = x * imgs_ratio[idx][1], height * imgs_ratio[idx][1]
            tmp.append([y, x, width, height])
        imgs_bbox[idx] = tmp

    imgs_class = [[ann[-1] for ann in trn_id_annotation[i]] for i in imgs_id]

    return trn_id_fname, trn_id_annotation, trn_id_single_anno, idx_category, category_idx, imgs, imgs_id, imgs_bbox, imgs_class

if __name__ == "__main__":
    trn_id_fname, trn_id_annotation, trn_id_single_anno, idx_category, category_idx, imgs, imgs_id, imgs_bbox, imgs_class = get_anno_data()
    print(trn_id_fname, trn_id_annotation, trn_id_single_anno, idx_category, category_idx, imgs)