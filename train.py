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
import torch.optim as optim
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
from model import get_SSD_model, lr_find
from VOC_data import VOC_dataset
from loss import *
from mAP import mAP

torch.set_printoptions(precision=3)

# PATH = 'C:\\datasets\\pascal\\'
# anno_path = f'{PATH}PASCAL_VOC\\pascal_train2007.json'
# train_dataset = VOC_dataset(PATH, anno_path)
# batch_size = 16
# learning_rate = 5e-4
# vgg_weight_path = 'C:\\Users\\ruifr\\.torch\\models\\vgg16-397923af.pth'


def detection_collate_fn(batch):
    imgs, bboxes, labels, img_id, ignore, img_scale = [], [], [], [], [], []
    for i, b, l, id, ig, s in batch:
        imgs.append(i)
        bboxes.append(b)
        labels.append(l)
        img_id.append(id)
        ignore.append(ig)
        img_scale.append(s)
    return torch.stack(imgs), bboxes, labels, img_id, ignore, img_scale


def adjust_lr(epoch):
    lr = 1
    if epoch > 80:
        lr /= 10
    if epoch > 100:
        lr /= 10
    return lr


def train_ssd():
    trn_id_fname, trn_id_annotation, trn_id_single_anno, idx_category, category_idx, imgs, imgs_id, imgs_bbox, imgs_class = (
        get_anno_data()
    )
    config = Config("remote")

    test_dataset = VOC_dataset(
        config.voc2007_root, config.voc2012_root, config.voc2007_test_anno, "test"
    )
    trn_dataset = VOC_dataset(
        config.voc2007_root, config.voc2012_root, config.anno_path, "trn"
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        collate_fn=detection_collate_fn,
    )
    trn_dataloader = DataLoader(
        trn_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        collate_fn=detection_collate_fn,
    )

    ssd_model = get_SSD_model(1, config.vgg_weight_path, config.vgg_reduced_weight_path)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    ssd_model = ssd_model.to(device)

    prior_box = get_prior_box()
    # prior_box = prior_box.to(device)

    loss_array = []
    val_array = []

    print("success build ssd model to train")
    optimizer = torch.optim.SGD(
        ssd_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5 * 1e-4
    )
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, adjust_lr)

    for epoch in range(120):
        lr_scheduler.step()

        # train
        for i, batch in enumerate(trn_dataloader):
            imgs, bboxes, labels, img_id, ignores, img_scale = batch
            # bboxes, labels, img_id, ignores, img_scale = bboxes[0], labels[0], img_id[0], ignores[0], img_scale[0]
            imgs = imgs.to(device)
            # bboxes = bboxes.to(device)
            cls_preds, loc_preds = ssd_model(imgs)

            ssd_model.zero_grad()
            total_loss = 0
            total_loc_loss, total_cls_loss = 0, 0

            for idx in range(imgs.shape[0]):

                img, bbox, label = imgs[idx], bboxes[idx], labels[idx]
                cls_pred, loc_pred = cls_preds[idx], loc_preds[idx]
                iou = get_iou(bbox, prior_box)

                pos_mask, cls_target, bbox_target = get_target(
                    iou, prior_box, img, bbox, label
                )
                pos_mask, cls_target, bbox_target = (
                    pos_mask.to(device),
                    cls_target.to(device),
                    bbox_target.to(device),
                )

                loss_loc, loss_cls = loss(
                    cls_pred, loc_pred, pos_mask, cls_target, bbox_target
                )
                total_loc_loss += loss_loc
                total_cls_loss += loss_cls

                total_loss += loss_cls + loss_loc

            total_loss /= float(imgs.shape[0])
            total_cls_loss /= float(imgs.shape[0])
            total_loc_loss /= float(imgs.shape[0])

            total_loss.backward()
            optimizer.step()

            cls_loss = round(float(total_cls_loss), 3)
            loc_loss = round(float(total_loc_loss), 3)
            t_loss = round(float(total_loss), 3)
            if i % 5 == 0:
                print(
                    epoch * 515 + i,
                    "cls_loss: {}, loc_loss: {}, loss: {}".format(
                        cls_loss, loc_loss, t_loss
                    ),
                )
            loss_array.append(t_loss)

        # val and save every 5 epoch
        if epoch % 5 == 0:
            torch.save(ssd_model.state_dict(), "f_trained_{}_epoch".format(i))
            print("val--------------------------")
            for val_i, batch in enumerate(test_dataloader):
                imgs, bboxes, labels, img_id, ignores, img_scale = batch
                imgs = imgs.to(device)
                cls_preds, loc_preds = ssd_model(imgs)

                total_loss = 0
                total_loc_loss, total_cls_loss = 0, 0

                for idx in range(imgs.shape[0]):
                    img, bbox, label = imgs[idx], bboxes[idx], labels[idx]
                    cls_pred, loc_pred = cls_preds[idx], loc_preds[idx]
                    iou = get_iou(bbox, prior_box)

                    pos_mask, cls_target, bbox_target = get_target(
                        iou, prior_box, img, bbox, label
                    )
                    pos_mask, cls_target, bbox_target = (
                        pos_mask.to(device),
                        cls_target.to(device),
                        bbox_target.to(device),
                    )

                    loss_loc, loss_cls = loss(
                        cls_pred, loc_pred, pos_mask, cls_target, bbox_target
                    )
                    total_loc_loss += loss_loc
                    total_cls_loss += loss_cls

                    total_loss += loss_cls + loss_loc

                total_loss /= float(imgs.shape[0])
                total_cls_loss /= float(imgs.shape[0])
                total_loc_loss /= float(imgs.shape[0])
                cls_loss = round(float(total_cls_loss), 3)

                loc_loss = round(float(total_loc_loss), 3)
                t_loss = round(float(total_loss), 3)
                if val_i % 100 == 0:
                    print(
                        val_i,
                        "cls_loss: {}, loc_loss: {}, loss: {}".format(
                            cls_loss, loc_loss, t_loss
                        ),
                    )
                val_array.append(t_loss)

            # valdiate the mAP
            print("mAP")
            mean_average_precision = mAP(config.voc2007_test_anno)
            for i, batch in tqdm_notebook(enumerate(test_dataloader)):
                imgs, bboxes, labels, img_id, ignores, img_scale = batch
                bboxes, labels, img_id, ignores, img_scale = (
                    bboxes[0],
                    labels[0],
                    img_id[0],
                    ignores[0],
                    img_scale[0],
                )
                imgs = imgs.to(device)
                cls_preds, loc_preds = ssd_model(imgs)

                res_score, res_bbox, res_cls = mean_average_precision.nms(
                    cls_preds, loc_preds, prior_box, conf_threshold=0.01
                )

                for _ in range(len(bboxes)):
                    bboxes[_][1] *= 300
                    bboxes[_][3] *= 300
                    bboxes[_][0] *= 300
                    bboxes[_][2] *= 300
                for _ in range(len(bboxes)):
                    mean_average_precision.add_single_gt(
                        img_id, bboxes[_], labels[_], ignores[_]
                    )

                for _ in range(len(res_score)):
                    mean_average_precision.add_single_pred(
                        img_id,
                        res_score[_].cpu().detach().numpy(),
                        res_bbox[_].cpu().detach().numpy(),
                        res_cls[_],
                    )
            res = mean_average_precision.calculate_mAP()

            # save loss array and map to log
            with open("res.json", "a") as f:
                json.dump({"epoch": epoch, "loss": loss_array, "map": res}, f)
                f.write("\n")
            print("mAP: ", np.mean([res[k] for k in res.keys()]))
            print("finish val--------------------------")


if __name__ == "__main__":
    train_ssd()
