import argparse
import math
import os
# import apex as amp
import random

import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import timm
from collections import OrderedDict


# import skcuda.linalg as linalg

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True,
                 mid_dim=256):
        super(ClassBlock, self).__init__()
        add_block = []
        self.Linear = nn.Linear(input_dim, num_bottleneck)
        self.bnorm = nn.BatchNorm1d(num_bottleneck)

        init.kaiming_normal_(self.Linear.weight.data, a=0, mode='fan_out')
        init.constant_(self.Linear.bias.data, 0.0)
        init.normal_(self.bnorm.weight.data, 1.0, 0.02)
        init.constant_(self.bnorm.bias.data, 0.0)

        classifier = []
        if droprate > 0:
            classifier += [nn.Dropout(p=droprate)]
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = classifier

    def forward(self, x):
        x = self.Linear(x)
        x = self.bnorm(x)
        x = self.classifier(x)
        return x


class CSWinTransv2_threeIn(nn.Module):

    def __init__(self, class_num, droprate, decouple, infonce, block=4):
        super(CSWinTransv2_threeIn, self).__init__()

        model_ft = timm.create_model('swinv2_base_window12to16_192to256_22kft1k', pretrained=True)
        model_ft.head = nn.Sequential()
        self.model = model_ft
        self.decouple = decouple
        self.infonce = infonce
        self.block = block
        self.pool = 'avg'
        self.bn = nn.BatchNorm1d(1024, affine=False)
        self.avg = nn.AdaptiveMaxPool2d((1, 1))
        self.infoAvg = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                     nn.BatchNorm2d(1024))
        for i in range(self.block):
            clas = 'classifier' + str(i + 1)
            setattr(self, clas, ClassBlock(2560, class_num, droprate=droprate))  # 2112 1440
        setattr(self, 'classifier0', ClassBlock(1536, class_num, droprate=droprate))
        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(512)

    def forward(self, x):
        B = x.shape[0]

        x = self.model.patch_embed(x)

        for blk in self.model.layers[0].blocks:
            x = blk(x)
        # 第一轮的2
        # y1 = x.transpose(2, 1)
        # y1 = y1.view(y1.size(0), y1.size(1), 64, 64)

        x = self.model.layers[0].downsample(x)

        for blk in self.model.layers[1].blocks:
            x = blk(x)
        # 第二轮的2
        y2 = x.transpose(2, 1)
        y2 = y2.view(y2.size(0), y2.size(1), 32, 32)

        x = self.model.layers[1].downsample(x)

        for element, blk in enumerate(self.model.layers[2].blocks):
            x = blk(x)
            if (element + 1) == 6:
                y31 = x.transpose(2, 1)
                y31 = y31.view(y31.size(0), y31.size(1), 16, 16)

            if (element + 1) == 12:
                y32 = x.transpose(2, 1)
                y32 = y32.view(y32.size(0), y32.size(1), 16, 16)

            if (element + 1) == 18:
                y33 = x.transpose(2, 1)
                y33 = y33.view(y33.size(0), y33.size(1), 16, 16)

        x = self.model.layers[2].downsample(x)

        x = self.model.norm(x)
        y4 = x.transpose(2, 1)
        y4 = y4.view(y4.size(0), y4.size(1), 8, 8)

        if self.infonce == 1:
            xi = y4

        # y4 = self.con4(y4)
        # 用y1y2y3y4去经过环形切割

        y2_ = self.get_part_pool(y31).view(y31.size(0), y31.size(1), -1)  # torch.Size([2, 192, 4]) 11
        y4_ = self.get_part_pool(y32).view(y32.size(0), y32.size(1), -1)  # torch.Size([2, 768, 4]) 11
        y5_ = self.get_part_pool(y33).view(y33.size(0), y33.size(1), -1)

        a, b, c, d = self.cat(y2_, y4_, y5_)  # abcd就是外环到内环的信息

        x = self.outpart(a, b, c, d)

        if self.infonce == 1:
            xi = self.infoAvg(xi)
            return [x, xi]
        else:
            return x

    def get_part_pool(self, x, no_overlap=True):
        result = []
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H / 2), int(W / 2)
        per_h, per_w = H / (2 * self.block), W / (2 * self.block)
        if per_h < 1 and per_w < 1:
            new_H, new_W = H + (self.block - c_h) * 2, W + (self.block - c_w) * 2
            x = nn.functional.interpolate(x, size=[new_H, new_W], mode='bilinear', align_corners=True)
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H / 2), int(W / 2)
            per_h, per_w = H / (2 * self.block), W / (2 * self.block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)
        for i in range(self.block):
            i = i + 1
            if i < self.block:
                x_curr = x[:, :, (c_h - i * per_h):(c_h + i * per_h), (c_w - i * per_w):(c_w + i * per_w)]
                x_pre = None
                if no_overlap and i > 1:
                    x_pre = x[:, :, (c_h - (i - 1) * per_h):(c_h + (i - 1) * per_h),
                            (c_w - (i - 1) * per_w):(c_w + (i - 1) * per_w)]
                    x_pad = F.pad(x_pre, (per_h, per_h, per_w, per_w), "constant", 0)
                    x_curr = x_curr - x_pad
                avgpool = self.avg_pool(x_curr, x_pre)
                result.append(avgpool)

            else:
                if no_overlap and i > 1:
                    x_pre = x[:, :, (c_h - (i - 1) * per_h):(c_h + (i - 1) * per_h),
                            (c_w - (i - 1) * per_w):(c_w + (i - 1) * per_w)]
                    pad_h = c_h - (i - 1) * per_h
                    pad_w = c_w - (i - 1) * per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2) + 2 * pad_h == H:
                        x_pad = F.pad(x_pre, (pad_h, pad_h, pad_w, pad_w), "constant", 0)
                    else:
                        ep = H - (x_pre.size(2) + 2 * pad_h)
                        x_pad = F.pad(x_pre, (pad_h + ep, pad_h, pad_w + ep, pad_w), "constant", 0)
                    x = x - x_pad
                avgpool = self.avg_pool(x, x_pre)
                result.append(avgpool)
        return torch.stack(result, dim=2)

    # 需要对y1234的长条进行加权并且进入分类器

    def cat(self, y1, y2, y3):

        for i in range(self.block):
            if i == 0:

                cat1 = torch.cat((y1[:, :, i], y2[:, :, i], y3[:, :, i],), dim=1)

            elif i == 1:

                cat2 = torch.cat((y1[:, :, i].view(y1.size(0), -1), y2[:, :, i].view(y2.size(0), -1),
                                  y3[:, :, i].view(y3.size(0), -1), y1[:, :, i - 1], y2[:, :, i - 1],), dim=1)
            elif i == 2:

                cat3 = torch.cat((y1[:, :, i].view(y1.size(0), -1), y2[:, :, i].view(y2.size(0), -1),
                                  y3[:, :, i].view(y3.size(0), -1), y1[:, :, i - 1], y2[:, :, i - 1],), dim=1)


            elif i == 3:
                # oneoneview
                cat4 = torch.cat((y1[:, :, i].view(y1.size(0), -1), y2[:, :, i].view(y2.size(0), -1),
                                  y3[:, :, i].view(y3.size(0), -1), y1[:, :, i - 1], y2[:, :, i - 1],), dim=1)

        return cat1, cat2, cat3, cat4

    def outpart(self, a, b, c1, d):
        out = []
        # for i in range(self.block-1):
        name = 'classifier0'
        c = getattr(self, name)
        out.append(c(a))

        name = 'classifier1'
        c = getattr(self, name)
        out.append(c(b))

        name = 'classifier2'
        c = getattr(self, name)
        out.append(c(c1))

        name = 'classifier3'
        c = getattr(self, name)
        out.append(c(d))

        if not self.training:
            return torch.stack(out, dim=2)
        else:
            return out

    def avg_pool(self, x_curr, x_pre=None):
        h, w = x_curr.size(2), x_curr.size(3)
        if x_pre == None:
            h_pre = w_pre = 0.0
        else:
            h_pre, w_pre = x_pre.size(2), x_pre.size(3)
        pix_num = h * w - h_pre * w_pre
        avg = x_curr.flatten(start_dim=2).sum(dim=2).div_(pix_num)
        return avg

    def part_class(self, y1, y2, y3, y4):
        out_p = []
        for i in range(self.block):
            name = 'classifier' + str(i)
            c = getattr(self, name)
            a = torch.cat((self.ln1(y1[:, :, i].view(y1.size(0), -1)), self.ln2(y2[:, :, i].view(y2.size(0), -1)),
                           self.ln3(y3[:, :, i].view(y3.size(0), -1)), self.ln4(y4[:, :, i].view(y4.size(0), -1))),
                          dim=1)

            out_p.append(c(a))
        if not self.training:
            return torch.stack(out_p, dim=2)
        else:
            return out_p

    def part_classifier(self, x):
        out_p = []
        for i in range(self.block):
            o_tmp = x[:, :, i].view(x.size(0), -1)
            # print(o_tmp.shape)
            name = 'classifier' + str(i)
            c = getattr(self, name)
            out_p.append(c(o_tmp))
            # print(out_p[0].shape, "out_p")
        if not self.training:
            return torch.stack(out_p, dim=2)
        else:
            return out_p

    def get_1x_lr_params(self):  # lr/10 learning rate
        modules = [self.model, ]
        for m in modules:
            yield from m.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.norm2, self.norm1,
                   getattr(self, 'classifier1'), getattr(self, 'classifier2'),
                   getattr(self, 'classifier3'), getattr(self, 'classifier0'), ]
        for m in modules:
            if isinstance(m, torch.nn.Parameter):
                yield m
            else:
                yield from m.parameters()
