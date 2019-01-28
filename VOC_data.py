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
from augmentation import SSDAugmentation

from Config import Config

torch.set_printoptions(precision=3)

class VOC_dataset(Dataset):
    def __init__(self, root_path, anno_path):
        '''
        Description:
        Dataset

        Arguments:
        root_path: (str) the path to data directory ('C:\\datasets\\pascal\\') in this case
        anno_path: (str) the path to annotation file ('C:\\datasets\\pascal\\PASCAL_VOC\\pascal_train2007.json') in this case
        '''
        self.dataset_json = json.load(open(anno_path))
#         self.img_path = root_path + '/JPEGImages/'
        self.img_path = root_path
        self.id_fname = {img['id']: img['file_name'] for img in self.dataset_json['images']}
        self.id_list = [k for k in self.id_fname.keys()]
    
        self.id_annotation = defaultdict(list)
        for anno in self.dataset_json['annotations']:
            self.id_annotation[anno['image_id']].append([anno['bbox'], anno['category_id']])

        self.id_single_anno = defaultdict(list)
        for anno in self.id_annotation:
            for bbox, c in self.id_annotation[anno]:
                if self.id_single_anno[anno] == [] or bbox[-1] * bbox[-2] > self.id_single_anno[anno][0][0][-1] * self.id_single_anno[anno][0][0][-2]:
                    self.id_single_anno[anno] = [[bbox, c]]

        self.idx_category = {tmp['id']: tmp['name'] for tmp in self.dataset_json['categories']}
        self.category_idx = {tmp['name']: tmp['id'] for tmp in self.dataset_json['categories']}

        self.transforms = SSDAugmentation()
    
    def __getitem__(self, idx):
        bbox, label = [], []
        
        for anno in self.id_annotation[self.id_list[idx]]:
            bbox.append(anno[0])
            label.append(anno[1])
        
        # print(self.img_path + self.id_fname[self.id_list[idx]])
        img = cv2.imread(self.img_path + self.id_fname[self.id_list[idx]])
        img, bbox, label = np.float32(img), np.float32(bbox).reshape(-1, 4), np.int32(label)
        
        # print(img.shape)
        
        img, bbox, label = self.transforms(img, bbox, label)

        img = torch.tensor(img).permute(2, 0, 1)

        return img, bbox, label
    
    def __len__(self):
        return len(self.dataset_json['images'])

if __name__ == "__main__":
    config = Config('remote')
    trn_dataset = VOC_dataset(config.voc2007_root, config.voc2007_trn_anno)
    img, bbox, label = trn_dataset[8]
    
    print(len(trn_dataset), img.shape, label.shape, bbox.shape)