import os
import torch
from tqdm import tqdm_notebook, tqdm
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

from Config import Config

torch.set_printoptions(precision=3)


class VOC_dataset(Dataset):
    def __init__(self, voc2007_root, voc2012_root, anno_path, trn_val="trn"):
        """
        Description:python
        Dataset Object for Pascal voc dataset

        Arguments:
        voc2007_root: (str), the path to voc 2007 image directory, e.g 'C:\\datasets\\pascal\\VOC2007\\JPEGImages\\' 
        voc2012_root: (str), the path to voc 2012 image directory, e.g 'C:\\datasets\\pascal\\VOC2012\\JPEGImages\\' 
        anno_path: List[(str)], the path to annotation file, in json format, order 12trn, 12val, 07trn, 07val

        trn_val: (str), trn | test, this dataset is for train or test
        """
        # self.trn_val = trn_val

        # build dataset trn, from voc2007 train + voc2007 val + voc2012 train + voc2012 val
        # all 4 part together, total len = 16551
        if trn_val == "trn":
            self.phase = "trn"
            voc2012_trn_anno, voc2012_val_anno, voc2007_trn_anno, voc2007_val_anno = (
                anno_path
            )
            self.dataset_json = [json.load(open(path)) for path in anno_path]
            self.root_path_dict = {
                0: voc2012_root,
                1: voc2012_root,
                2: voc2007_root,
                3: voc2007_root,
            }

            self.id_fname = dict()
            for idx in range(4):
                for img in self.dataset_json[idx]["images"]:
                    self.id_fname[img["id"]] = (
                        self.root_path_dict[idx] + img["file_name"]
                    )

            self.id_list = [k for k in self.id_fname.keys()]

            self.id_annotation = defaultdict(list)
            for idx in range(4):
                for anno in self.dataset_json[idx]["annotations"]:
                    self.id_annotation[anno["image_id"]].append(
                        [anno["bbox"], anno["category_id"], anno["ignore"]]
                    )

            self.transforms = SSDAugmentation()
            self.idx_category = {
                tmp["id"]: tmp["name"]
                for tmp in self.dataset_json[idx]["categories"]
                for idx in range(4)
            }
            self.category_idx = {
                tmp["name"]: tmp["id"]
                for tmp in self.dataset_json[idx]["categories"]
                for idx in range(4)
            }

        # build dataset test, voc2007 test
        else:
            self.phase = "test"
            self.dataset_json = json.load(open(anno_path))
            self.id_fname = {
                img["id"]: voc2007_root + img["file_name"]
                for img in self.dataset_json["images"]
            }
            self.id_list = [k for k in self.id_fname.keys()]
            self.id_annotation = defaultdict(list)
            for anno in self.dataset_json["annotations"]:
                self.id_annotation[anno["image_id"]].append(
                    [anno["bbox"], anno["category_id"], anno["ignore"]]
                )
            self.transforms = SSD_Val_Augmentation()
            self.idx_category = {
                tmp["id"]: tmp["name"] for tmp in self.dataset_json["categories"]
            }
            self.category_idx = {
                tmp["name"]: tmp["id"] for tmp in self.dataset_json["categories"]
            }

    def __getitem__(self, idx):
        bbox, label, ignore = [], [], []

        for anno in self.id_annotation[self.id_list[idx]]:
            bbox.append(anno[0])
            label.append(anno[1])
            ignore.append(anno[2])

        # print(self.img_path + self.id_fname[self.id_list[idx]])
        img_id = self.id_list[idx]
        img = cv2.imread(self.id_fname[self.id_list[idx]])
        img, bbox, label = (
            np.float32(img),
            np.float32(bbox).reshape(-1, 4),
            np.int32(label),
        )

        # get image scale
        img_scale = img.shape[0], img.shape[1]

        # print(img.shape)
        img, bbox, label = self.transforms(img, bbox, label)

        # from (h, w, c) to (c, h, w)
        img = torch.tensor(img).permute(2, 0, 1)

        # from bgr to rgb
        img = img[(2, 1, 0), :, :]

        return img, bbox, label, img_id, ignore, img_scale

    def __len__(self):
        return len(self.id_list)


if __name__ == "__main__":
    config = Config("local")
    train_dataset = VOC_dataset(
        config.voc2007_root,
        config.voc2012_root,
        config.voc2007_test_anno,
        trn_val="test",
    )
    # just check everything is fine
    for img, bbox, label in tqdm(train_dataset):
        pass
    print(len(train_dataset))
