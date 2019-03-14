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
from SSDloss import *

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
        loc_ = decode(loc[0], priors, [0.1, 0.2]) * 300

        conf_ = F.softmax(conf[0])

        # ignore the bkg class
        # conf_score, conf_cls = torch.max(conf_[:, 1:], dim=1)
        conf_score, conf_cls = torch.max(conf_, dim=1)
        conf_bkg_mask = conf_cls != 0
        conf_score, conf_cls, loc_ = conf_score[conf_bkg_mask], conf_cls[conf_bkg_mask], loc_[conf_bkg_mask]

        #     if use trained model, add this line
        if use_trained_model:
            conf_cls += 1

        conf_mask = conf_score > conf_threshold
        conf_score, conf_cls, loc_ = conf_score[conf_mask], conf_cls[conf_mask], loc_[conf_mask]

        conf_score, conf_idx = torch.sort(conf_score, descending=True)
        conf_cls, bboxes = conf_cls[conf_idx], loc_[conf_idx]

        wh = bboxes[:, 2:] - bboxes[:, :2]
        areaes = wh[:, 0] * wh[:, 1]

        # 2. get the result bbox and result class
        res_score, res_bbox, res_cls = [], [], []

        # how many detections for each class could been found
        cls_count = {_: 200 for _ in range(1, 21)}

        for idx in range(len(conf_score)):
            if conf_score[idx] != 0:
                res_score.append(conf_score[idx])
                res_bbox.append(bboxes[idx])
                res_cls.append(conf_cls[idx])

                cur_class = int(conf_cls[idx])
                for i_head in range(idx + 1, len(conf_score)):
                    if conf_cls[i_head] != cur_class:
                        continue

                    else:
                        if conf_score[i_head] != 0:
                            max_xy = torch.max(bboxes[idx][:2], bboxes[i_head][:2])
                            min_xy = torch.min(bboxes[idx][2:], bboxes[i_head][2:])
                            wh = torch.clamp(min_xy - max_xy, min=0)
                            intersect = wh[0] * wh[1]
                            iou = intersect / (areaes[idx] + areaes[i_head] - intersect)
                            if iou > iou_threshold and cls_count[cur_class] > 0:
                                cls_count[cur_class] -= 1
                                conf_score[i_head] = 0

        # no need to restrict top k
        # res_score, res_bbox, res_cls = res_score[:top_k], res_bbox[:top_k], res_cls[:top_k]
        
        new_res_score, new_res_bbox, new_res_cls = [], [], []
        for i in range(len(res_score)):
            if res_score[i] > 0.6:
                new_res_score.append(res_score[i])
                new_res_bbox.append(res_bbox[i])
                new_res_cls.append(res_cls[i])
                
#         return res_score, res_bbox, res_cls
        return new_res_score, new_res_bbox, new_res_cls

if __name__ == "__main__":
    config = Config('local')
    ssd_model = get_SSD_model(config.batch_size, config.vgg_weight_path, config.vgg_reduced_weight_path)
    ssd_model.load_trained_model(config.trained_path)

    test_dataset = VOC_dataset(config.voc2007_root, config.voc2012_root, config.voc2007_test_anno, 'test')
    # DataLoader = DataLoader(test_dataset)

    for idx in range(15):
        # idx = 19
        img, bbox, label, img_id, ignore, img_scale = test_dataset[idx]
        
        mean_average_precision = mAP(config.voc2007_test_anno)
        # print(bbox, '\n', label, '\n', img_id, '\n', ignore)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        ssd_model = ssd_model.to(device)
        conf, loc = ssd_model(img.unsqueeze(0).to(device))
        priors = get_prior_box()

        res_score, res_bbox, res_cls = mean_average_precision.nms(conf, loc, priors.to(device))
        print('-----------------{}-----------------'.format(idx))
        for _ in range(len(res_bbox)):
            res_bbox[_][1] *= img_scale[0] / 300
            res_bbox[_][3] *= img_scale[0] / 300
            res_bbox[_][0] *= img_scale[1] / 300
            res_bbox[_][2] *= img_scale[1] / 300
            print('bbox: {}, {}'.format(_, res_bbox[_].cpu().detach().numpy()))
            print('class: {}, {}'.format(_, int(res_cls[_])))
        # print(idx, '\n', res_bbox, '\n', res_cls, '\n', res_score)

    # config = Config('local')
    # mean_average_precision = mAP(config.voc2007_test_anno)
    # test1 = None
    # with open('./test1.txt') as f:
    #     test1 = f.readlines()

    # img_id = -1

    # for i, line in tqdm_notebook(enumerate(test1)):
    #     if line == '\n':
    #         continue
    #     if line.split(':')[0] == 'GROUND TRUTH FOR':
    #         img_id = int(line.split(':')[-1])
    #     elif line.split(':')[0] == 'label':
    #         xyxy = line.split(':')[1].split('||')[:4]
    #         xyxy = np.array([float(i) for i in xyxy])
    # #         print(line)
    #         label = int(line.split(':')[1].split('||')[-1])
            
    #         mean_average_precision.add_single_gt(img_id, xyxy, label + 1, 0)
    #     elif line.split(':')[0] == 'PREDICTIONS':
    #         continue
    #     else:
    # #         print(line.split(':')[2].split(' ')[1])
    #         label = catagory_idx[line.split(':')[1].split(' ')[1]]
    #         score = float(line.split(':')[2].split(' ')[1][7: -1])
    #         xyxy = line.split(':')[-1].split(')')[-1].split('||')
    #         xyxy = np.array([float(i) for i in xyxy])
            
    #         mean_average_precision.add_single_pred(img_id, score, xyxy, label)
    
    # # mean_average_precision.get_gt_num()
    # mAP = mean_average_precision.calculate_mAP()
    # print(mAP)

    # --------------------------------------

    # config = Config('local')
    # mean_average_precision = mAP(config.voc2007_test_anno)

    # id_anno = mean_average_precision.get_id_annotation(config.voc2007_test_anno)

    # for img_id in id_anno.keys():
    #     for anno in id_anno[img_id]:
    #         bbox = np.array(anno['bbox'])
    #         bbox[2:] = bbox[:2] + bbox[2:]
    #         mean_average_precision.add_single_gt(img_id, bbox, anno['label'], anno['ignore'])
            
    # all_file = glob.glob('./results/*.txt')
    # for file in all_file:
    #     idx = catagory_idx[file.split('_')[2].split('.')[0]]
    #     with open(file) as f:
    #         lines = f.readlines()
    #         for line in lines:
    #             line = line.split(' ')
    #             img_id = int(line[0])
    #             score = float(line[1])
    #             bbox = np.array([float(i) for i in line[2:]])
    #             mean_average_precision.add_single_pred(img_id, score, bbox, idx)

    # mAP = mean_average_precision.calculate_mAP(metric='12')
    
    # print(mAP)
    # print(np.mean([mAP[k] for k in mAP.keys()]))