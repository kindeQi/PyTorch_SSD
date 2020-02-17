# refer: https://gluon-cv.mxnet.io/_modules/gluoncv/utils/metrics/voc_detection.html
# refer: https://github.com/amdegroot/ssd.pytorch/blob/master/eval.py
# refer: https://github.com/rafaelpadilla/Object-Detection-Metrics

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

# from fastai import transforms, model, dataset, conv_learner

from PIL import ImageDraw, ImageFont
from matplotlib import patches, patheffects
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from augmentation import SSDAugmentation, SSD_Val_Augmentation

from Config import *
from SSD_model import get_SSD_model, lr_find
from VOC_data import VOC_dataset
from loss import *

from pathlib import Path
import visdom
import sys
import time
import random
import datetime
import glob

class mAP(object):
    def __init__(self, test_json_path):
        self.id_anno = self.get_id_annotation(test_json_path)
        self.idx_catagory = idx_catagory
        self.catagory_idx = catagory_idx
        
        '''
        self.detection_res.keys() = [1, 20], corresponding to catagory [1, 20]
        self.detection_res[key1] = List[{'score':, 'bbox':, 'label':, 'img_id':} ...]
        len(self.detection_res[key1]) == the number of detected bbox for class key1
        '''
        self.detection_res = {i: [] for i in range(1, 21)}
    
        '''
        self.ground_truths.keys() = all id of images in voc2007 test dataset
        slef.ground_truths[img_id1] = List[Dict{'bbox':, 'label':, 'used':False} ...]
        len(self.ground_truths) = 4952 (number of images in voc2007 test dataset)
        '''
        self.ground_truths = {i: [] for i in self.id_anno.keys()}
    

    def get_id_annotation(self, json_path):
        '''
        Description:
        return the id-annotation dict
        e.g. id_anno[1], return the bboxes, labels, ignore of the image
        
        Arguments:
        json_path: path to the json file
        
        Return:
        id_annotation file
        '''
        assert os.path.exists(json_path)
        json_file = json.load(open(json_path, 'r'))
        
        id_annotation = defaultdict(list)
        for anno in json_file['annotations']:
            id_annotation[anno['image_id']].append({'bbox': anno['bbox'], 'label': anno['category_id'], 'ignore': anno['ignore']})
        return id_annotation


    def iou(self, bbox1, bbox2):
        wh1 = bbox1[2:] - bbox1[:2]
        wh2 = bbox2[2:] - bbox2[:2]
        area1 = wh1[0] * wh1[1]
        area2 = wh2[0] * wh2[1]
        
        max_xy = np.max(np.stack([bbox1[:2], bbox2[:2]]), axis=0)
        min_xy = np.min(np.stack([bbox1[2:], bbox2[2:]]), axis=0)
        wh  = np.clip(min_xy - max_xy, a_min=0, a_max=float('inf'))
        
        intersect = wh[0] * wh[1]
        iou = intersect / (area1 + area2 - intersect)
        
        return iou
    

    def get_gt_num(self, ignore=True):
        '''
        ignore: ignore the difficult images or not
        '''
        self.gt_num = {_: 0 for _ in range(1, 21)}
        for img_id in self.ground_truths.keys():
            for anno in self.ground_truths[img_id]:
                if not anno['ignore']:
                    self.gt_num[anno['label']] += 1
    
    
    def calculate_AP(self, recall, precision, metric='07'):
        if metric == '07':
            ap = 0
            for t in np.arange(0, 1.1, 0.1):
                if np.sum(recall > t) == 0:
                    p = 0
                else:
                    p = np.max(precision[recall > t])
                ap += p / 11
        
        else:
            ap = 0
            prev_recall = 0
            for r in recall:
                if np.sum(recall > r) == 0:
                    continue
                p = np.max(precision[recall > r])
                ap += p * (r - prev_recall)
                prev_recall = r

        return ap      
            
    
    def calculate_mAP(self, iou_threshold=0.5, metric='12'):
        '''
        metric: 07|12
        07 use 11 interpolation, 12 use all points interpolation
        '''
        self.get_gt_num()
        mAP = dict()
        
        for cls in range(1, 21):
            det = self.detection_res[cls]
            gt =  self.ground_truths
            
            tp, fp = np.array([0 for _ in range(len(det))]), np.array([0 for _ in range(len(det))])
            det = sorted(det, key=lambda k: k['score'], reverse=True)
            
            for det_i, det_item in enumerate(det):
                img_id = det_item['img_id']
                bbox1, label1 = det_item['bbox'], det_item['label']
                
                matched = False
                for i, anno in enumerate(gt[img_id]):
                    if matched:
                        break
                    if anno['used'] or label1 != anno['label'] or anno['ignore']:
                        continue
                    iou = self.iou(bbox1, anno['bbox'])
                    if iou > iou_threshold:
                        tp[det_i:] += 1 
                        gt[img_id][i]['used'] = True
                        matched = True
                if not matched:
                    fp[det_i:] += 1
                    
            recall = tp / self.gt_num[cls]
            precision = tp / (tp + fp)
            ap = self.calculate_AP(recall, precision, metric)
            mAP[cls] = ap

        return mAP


    def add_groundtruth(self, img_id, bboxes, labels, ignores):
        self.ground_truths[img_id] = [{'bbox': bbox, 'label': label, 'used': False, 'ignore': ignore} 
                                      for bbox, label, ignore in zip(np.array(bboxes), 
                                                                   np.array(labels),
                                                                   np.array(ignores))]

    def add_predictions(self, img_id, conf_threshold, iou_threshold, top_k, conf, loc, priors):
        nms_score, nms_bbox, nms_cls = self.nms(conf_threshold, iou_threshold, top_k, conf, loc, priors)
        for score, bbox, cls in zip(nms_score, nms_bbox, nms_cls):
            self.detection_res[int(cls)].append({'score': score, 'bbox': bbox, 'label': cls, 'img_id':img_id})
    
    def add_single_gt(self, img_id, bbox, label, ignore):
        self.ground_truths[img_id].append({'bbox': np.array(bbox), 'label': label, 'used': False, 'ignore': ignore})
    
    def add_single_pred(self, img_id, score, bbox, cls):
        self.detection_res[int(cls)].append({'score': score, 'bbox': np.array(bbox), 'label': cls, 'img_id':img_id})

    

    def nms(self, conf, loc, priors, conf_threshold=0.01, iou_threshold=0.45, top_k=200, use_trained_model=True):
        '''
        Description:
        greedy nms

        Arguments:
        conf_threshold: int, default=0.45
        iou_threshold: int, defualt=0.01
        top_k: int, default=200
        conf: 
        loc:
        priors
        '''
        # 1. get the conf_score, conf_cls, bboxes and areaes
        loc = loc.cpu()
        priors = priors.cpu()
        loc_ = decode(loc[0], priors, [0.1, 0.2]) * 300

        conf_ = F.softmax(conf[0])

        # ignore the bkg class
        conf_score, conf_cls = torch.max(conf_[:, 1:], dim=1)
        if use_trained_model:
            conf_cls += 1

        conf_mask = conf_score > conf_threshold
        conf_score, conf_cls, loc_ = conf_score[conf_mask], conf_cls[conf_mask], loc_[conf_mask]

        conf_score, conf_idx = torch.sort(conf_score, descending=True)
        conf_cls, bboxes = conf_cls[conf_idx], loc_[conf_idx]

        res_score, res_bbox, res_cls = [], [], []

        # keep top 200 results for each class
        conf_score, conf_cls, bboxes = conf_score[:top_k * 3], conf_cls[:top_k * 3], bboxes[:top_k * 3]

        for class_idx in range(1, 21):
            class_mask = (conf_cls == class_idx)
            if torch.sum(class_mask) == 0:
                continue
            # else:
            #     print(class_idx, torch.sum(class_mask))
            conf_score_, bboxes_ = conf_score[class_mask][:200], bboxes[class_mask][:200]

            wh = bboxes_[:, 2:] - bboxes_[:, :2]
            areaes = wh[:, 0] * wh[:, 1]

            while len(conf_score_) > 0:
                cur_bbox = bboxes_[0]
                cur_score = conf_score_[0]
                cur_area = areaes[0]

                res_score.append(cur_score)
                res_bbox.append(cur_bbox)
                res_cls.append(class_idx)

                conf_score_ = conf_score_[1:]
                bboxes_ = bboxes_[1:]
                areaes = areaes[1:]        

                if len(conf_score_) == 0:
                    break        

                max_x1 = torch.clamp(bboxes_[:, 0], min=float(cur_bbox[0]))
                max_y1 = torch.clamp(bboxes_[:, 1], min=float(cur_bbox[1]))
                min_x2 = torch.clamp(bboxes_[:, 2], max=float(cur_bbox[2]))
                min_y2 = torch.clamp(bboxes_[:, 3], max=float(cur_bbox[3]))

                w = torch.clamp(min_x2 - max_x1, min=0)
                h = torch.clamp(min_y2 - max_y1, min=0)

                intercests = w * h
                iou = intercests / (cur_area + areaes - intercests) 
                iou_mask = iou < 0.45

                bboxes_ = bboxes_[iou_mask]
                conf_score_ = conf_score_[iou_mask]
                areaes = areaes[iou_mask]
                
        return res_score, res_bbox, res_cls


if __name__ == "__main__":
    config = Config('local')
    ssd_model = get_SSD_model(config.batch_size, config.vgg_weight_path, config.vgg_reduced_weight_path)
    ssd_model.load_trained_model(config.trained_path)

    test_dataset = VOC_dataset(config.voc2007_root, config.voc2012_root, config.voc2007_test_anno, 'test')
    # DataLoader = DataLoader(test_dataset)

    # for idx in range(15):
    idx = 3
    img, bbox, label, img_id, ignore, img_scale = test_dataset[idx]
    
    mean_average_precision = mAP(config.voc2007_test_anno)
    # print(bbox, '\n', label, '\n', img_id, '\n', ignore)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    ssd_model = ssd_model.to(device)
    conf, loc = ssd_model(img.unsqueeze(0).to(device))
    priors = get_prior_box()

    res_score, res_bbox, res_cls = mean_average_precision.nms(conf, loc, priors.to(device), conf_threshold=0.01)
    print('-----------------{}-----------------'.format(img_id))
    print("ground truths:\n")
    for _ in range(len(bbox)):
        bbox[_][1] *= img_scale[0]
        bbox[_][3] *= img_scale[0]
        bbox[_][0] *= img_scale[1]
        bbox[_][2] *= img_scale[1]
        print("gt_bbox: {}, {}".format(_ + 1, bbox[_]))
    print('labels: {}\n'.format(label))
    
    print("predictions: \n")
    for _ in range(len(res_bbox)):
        res_bbox[_][1] *= img_scale[0] / 300
        res_bbox[_][3] *= img_scale[0] / 300
        res_bbox[_][0] *= img_scale[1] / 300
        res_bbox[_][2] *= img_scale[1] / 300
        print('bbox: {}, {}'.format(_ + 1, res_bbox[_].cpu().detach().numpy()))
        print('class: {}, {}'.format(int(res_cls[_]), res_score[_]))
