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
from tqdm import tqdm

# from fastai import transforms, model, dataset, conv_learner

from PIL import ImageDraw, ImageFont
from matplotlib import patches, patheffects
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

torch.set_printoptions(precision=3)

from VOC_data import VOC_dataset
from Config import Config

# from draw_img_utils import *
from loss import *


def detection_collate_fn(batch):
    imgs, bboxes, labels = [], [], []
    for i, b, l in batch:
        imgs.append(i)
        bboxes.append(b)
        labels.append(l)
    return torch.stack(imgs), bboxes, labels


class L2norm(nn.Module):
    def __init__(self, n_channels, gamma):
        """
        Description:
        L2 norm layer

        Arguemnt
        n_channels: (int), number of channels get into the layer
        gamma: gamma parameter for norm
        """
        super(L2norm, self).__init__()
        self.gamma = gamma
        self.weight = nn.Parameter(torch.randn(n_channels))
        self.weight = nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = torch.sum(x ** 2, dim=1, keepdim=True) ** 0.5
        weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        return weight * x / norm


class SSD(nn.Module):
    def __init__(
        self,
        batch_size,
        base_net,
        reduced_fc,
        extra,
        conf_layers,
        loc_layers,
        special_layers,
        l2_norm,
    ):
        super(SSD, self).__init__()
        self.base_net = nn.ModuleList(base_net)
        self.reduced_fc = nn.ModuleList(reduced_fc)
        self.extra = nn.ModuleList(extra)
        self.conf_layers = nn.ModuleList(conf_layers)
        self.loc_layers = nn.ModuleList(loc_layers)
        self.l2_norm = l2_norm

        self.special_layers = special_layers
        self.batch_size = batch_size

    def forward(self, x):
        self.conf_res, self.loc_res = [], []
        self.conf, self.loc = [], []

        # base net
        for i, l in enumerate(self.base_net):
            x = l(x)

            # conv4_3
            if i == 22:
                x_l2_norm = self.l2_norm(x)
                self.conf_res.append(self.conf_layers[0](x_l2_norm))
                self.loc_res.append(self.loc_layers[0](x_l2_norm))
        # print(0, torch.sum(x_l2_norm))

        # reduced_fc
        for l in self.reduced_fc:
            x = l(x)

        self.conf_res.append(self.conf_layers[1](x))
        self.loc_res.append(self.loc_layers[1](x))
        # print(1, torch.sum(x))

        # extra_layer
        for i, l in enumerate(self.extra):
            # x = l(x)
            x = F.relu(l(x), inplace=True)
            if i % 2 == 1:
                self.conf_res.append(self.conf_layers[i // 2 + 2](x))
                self.loc_res.append(self.loc_layers[i // 2 + 2](x))

        self.conf = torch.cat(
            [
                l.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 21)
                for l in self.conf_res
            ],
            dim=1,
        )
        self.loc = torch.cat(
            [
                l.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 4)
                for l in self.loc_res
            ],
            dim=1,
        )

        return self.conf, self.loc

    def freeze_basenet(self):
        for l in self.base_net.parameters():
            l.requires_grad = False

    def defreeze_basenet(self):
        for l in self.base_net.parameters():
            l.requires_grad = True

    def load_reduced_fc_weight(self, reduced_fc_weight_path):
        reduced_fc_weight = torch.load(reduced_fc_weight_path)
        ssd_weight = self.state_dict()

        for k0, k1 in zip(ssd_weight, reduced_fc_weight):
            # print(k0, k1)
            ssd_weight[k0] = reduced_fc_weight[k1]

        self.load_state_dict(ssd_weight)

    def load_trained_model(self, trained_path):
        original_state_dict = self.state_dict()
        trained_state_dict = torch.load(trained_path)

        trained2original = {
            "vgg": "base_net",
            "L2Norm": "l2_norm",
            "extras": "extra",
            "loc": "loc_layers",
            "conf": "conf_layers",
        }

        for trained_k in trained_state_dict.keys():
            layer_name = trained_k.split(".")[0]
            layer_num = trained_k.split(".")[1]
            layer_other = ".".join((trained_k.split(".")[2:]))
            if layer_name == "L2Norm":
                original_key = "l2_norm.weight"
            #         original_state_dict['l2_norm.weights'] = trained_state_dict[k]
            elif layer_name == "vgg" and int(layer_num) > 28:
                num = str(int(layer_num) - 30)
                original_key = ".".join(["reduced_fc", num, layer_other])
            else:
                original_key = ".".join(
                    [trained2original[layer_name], layer_num, layer_other]
                )
            #     print('{:15s}:{:15s}'.format(trained_k, original_key))
            original_state_dict[original_key] = trained_state_dict[trained_k]

        self.load_state_dict(original_state_dict)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # nn.init.xavier_uniform(m.weight.data)
        nn.init.kaiming_uniform(m.weight.data)
        m.bias.data.zero_()


def lr_find(model, lr_max, lr_min, trn_dataloader, linear=True):
    """

    """
    torch.save(model.state_dict(), "tmp.pth")
    lr_array, loss_array = [], []

    num_batch = len(trn_dataloader)
    ratio = lr_max / lr_min
    step = (lr_max - lr_min) / num_batch

    optimizer = torch.optim.SGD(model.parameters(), lr=lr_min, momentum=0.9)
    print("SGD")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model.freeze_basenet()
    model = model.to(device)
    prior_box = get_prior_box()
    min_loss = float("inf")

    for i, batch in enumerate(tqdm(trn_dataloader)):
        if linear:
            lr = lr_min + step * i
        else:
            lr = lr_min * ratio ** (i / num_batch)
        optimizer.param_groups[0]["lr"] = lr

        imgs, bboxes, labels = batch
        imgs = imgs.to(device)
        cls_preds, loc_preds = model(imgs)

        model.zero_grad()

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

        # use min_loss to terminate the lr find more quickly
        if min_loss * 4 <= float(total_loss):
            break
        min_loss = min(float(min_loss), float(total_loss.data))
        # print(min_loss)

        total_loss.backward()
        optimizer.step()

        lr_array.append(lr)
        loss_array.append(round(float(total_loss), 3))

    model.state_dict = torch.load("tmp.pth")
    return lr_array, loss_array


def get_SSD_model(batch_size, vgg_weight_path, reduced_fc_weight):
    norm = L2norm(512, 20)

    weights_path = vgg_weight_path

    model = models.vgg16()
    model.load_state_dict(torch.load(weights_path))
    vgg_basenet = model.features[:30]

    # pool5, conv6, conv7
    reduced_fc = [
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
        nn.ReLU(inplace=True),
        nn.Conv2d(1024, 1024, kernel_size=1),
        nn.ReLU(inplace=True),
    ]

    extra = [
        nn.Conv2d(1024, 256, kernel_size=1, stride=1),
        nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        nn.Conv2d(512, 128, kernel_size=1, stride=1),
        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        nn.Conv2d(256, 128, kernel_size=1, stride=1),
        nn.Conv2d(128, 256, kernel_size=3, stride=1),
        nn.Conv2d(256, 128, kernel_size=1, stride=1),
        nn.Conv2d(128, 256, kernel_size=3, stride=1),
    ]

    # base_net = nn.ModuleList(list(vgg_basenet.children()) + list(extra.children()))
    # base_net = list(base_net.children())

    base_net = list(vgg_basenet.children())
    base_net[16].ceil_mode = True

    special_layers = [1, 3, 5, 7]

    conf_layers = [
        nn.Conv2d(512, 84, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(1024, 126, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(512, 126, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(256, 126, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(256, 84, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(256, 84, kernel_size=3, stride=1, padding=1),
    ]

    loc_layers = [
        nn.Conv2d(512, 16, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(1024, 24, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(512, 24, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(256, 24, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1),
    ]
    model = SSD(
        batch_size,
        base_net,
        reduced_fc,
        extra,
        conf_layers,
        loc_layers,
        special_layers,
        norm,
    )

    # init the model
    model.reduced_fc.apply(weights_init)
    model.extra.apply(weights_init)
    model.conf_layers.apply(weights_init)
    model.loc_layers.apply(weights_init)

    # use reduced fc weight instead of the original vgg weight + xavier init conv6, 7
    model.load_reduced_fc_weight(reduced_fc_weight)

    return model


if __name__ == "__main__":

    config = Config("local")
    ssd_model = get_SSD_model(1, config.vgg_weight_path, config.vgg_reduced_weight_path)
    # ssd_model.freeze_basenet()

    print(ssd_model.base_net[0].bias)

    print("success build ssd model")

    train_dataset = VOC_dataset(config.voc2007_root, config.voc2007_trn_anno)
    ssd_model.load_trained_model(config.trained_path)
