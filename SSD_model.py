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

torch.set_printoptions(precision=3)

# from SSD_model import get_SSD_model
from VOC_data import VOC_dataset
from Config import Config
# from draw_img_utils import *
# from SSDloss import *

class L2norm(nn.Module):
    def __init__(self, n_channels, gamma):
        '''
        Description:
        L2 norm layer

        Arguemnt
        n_channels: (int), number of channels get into the layer
        gamma: gamma parameter for norm
        '''
        super(L2norm, self).__init__()
        self.gamma = gamma
        self.weights = nn.Parameter(torch.randn(n_channels))
        self.weights = nn.init.constant_(self.weights, self.gamma)
    
    def forward(self, x):
        norm = torch.sum(x ** 2, dim=1, keepdim=True) ** 0.5
        weights = self.weights.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    
        return weights * x / norm

class SSD(nn.Module):
    def __init__(self, batch_size, base_net, reduced_fc, extra, conf_layers, loc_layers, special_layers, l2_norm):
        super(SSD, self).__init__()
        self.base_net = nn.ModuleList(base_net)
        self.reduced_fc = nn.ModuleList(reduced_fc)
        self.extra = nn.ModuleList(extra)
        self.conf_layers = nn.ModuleList(conf_layers)
        self.loc_layers = nn.ModuleList(loc_layers)
        self.l2_norm = l2_norm

        # self.base_net = base_net
        # self.reduced_fc = reduced_fc
        # self.extra = extra
        # self.conf_layers = conf_layers
        # self.loc_layers = loc_layers
        # self.l2_norm = l2_norm

        self.special_layers = special_layers
        self.batch_size = batch_size
        
    def forward(self, x):
        # test purpose
        # start = 0
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

        # reduced_fc
        for l in self.reduced_fc:
            x = l(x)
        
        self.conf_res.append(self.conf_layers[1](x))
        self.loc_res.append(self.loc_layers[1](x))

        #extra_layer
        for i, l in enumerate(self.extra):
            x = l(x)
            if i % 2 == 1:
                self.conf_res.append(self.conf_layers[i // 2 + 2](x))
                self.loc_res.append(self.loc_layers[i // 2 + 2](x))

#         for k, end_layer in enumerate(self.special_layers):
# #             print(start, end_layer)
#             for l in self.base_net[start: end_layer + 1]:
# #                 print(x.shape)
#                 x = l(x)
#             self.conf_res.append(self.conf_layers[k](x))
#             self.loc_res.append(self.loc_layers[k](x))
            
#             start = end_layer + 1
# #             print(k, x.shape)
        
        self.conf = torch.cat([l.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 21) for l in self.conf_res], dim=1)
        self.loc = torch.cat([l.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 4) for l in self.loc_res], dim=1)
        
        return self.conf, self.loc

    def freeze_basenet(self):
        for l in self.base_net.parameters():
            l.requires_grad = False

    def defreeze_basenet(self):
        for l in self.base_net.parameters():
            l.requires_grad = True

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight.data)
        m.bias.data.zero_()


def get_SSD_model(batch_size, vgg_weight_path):
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
        nn.Conv2d(128, 256, kernel_size=3, stride=1)
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
    model = SSD(batch_size, base_net, reduced_fc, extra, conf_layers, loc_layers, special_layers, norm)

    # init the model
    model.reduced_fc.apply(weights_init)
    model.extra.apply(weights_init)
    model.conf_layers.apply(weights_init)
    model.loc_layers.apply(weights_init)

    return model

if __name__ == "__main__":

    config = Config('local')
    ssd_model = get_SSD_model(1, config.vgg_weight_path)
    ssd_model.freeze_basenet()

    print('success build ssd model')

    train_dataset = VOC_dataset(config.voc2007_root, config.voc2007_trn_anno)

    img, bbox, label = train_dataset[0]
    img = img.unsqueeze(0)

    conf_pred, loc_pred = ssd_model(img)
    print(conf_pred.shape, loc_pred.shape)