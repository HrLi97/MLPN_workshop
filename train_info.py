# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse

# import neptune
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib
import random
import scipy.io

matplotlib.use('agg')
import matplotlib.pyplot as plt
# from PIL import Image
import copy
import time
import os
from model import *
from random_erasing import RandomErasing
from autoaugment import ImageNetPolicy, CIFAR10Policy
import yaml
import math
from shutil import copyfile
from utils import *
import numpy as np
from image_folder import SatData, DroneData, ImageFolder_selectID, ImageFolder_expandID, customData
import wandb

# print(torch.version.cuda,"torch.version.cuda")
# 11.1 torch.version.cuda

version = torch.__version__
# fp16
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError:  # will be 3.x series
    print(
        'This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name', default='debug', type=str, help='output model name')
parser.add_argument('--pool', default='avg', type=str, help='pool avg')
parser.add_argument('--data_dir', default='/home/lihaoran/BJDD_datesets/datesets/University-Release/train', type=str,
                    help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
parser.add_argument('--stride', default=1, type=int, help='stride')
parser.add_argument('--pad', default=10, type=int, help='padding')
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')
parser.add_argument('--views', default=2, type=int, help='the number of views')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--use_NAS', action='store_true', help='use NAS')
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--moving_avg', default=1.0, type=float, help='moving average')
parser.add_argument('--droprate', default=0.75, type=float, help='drop rate')
parser.add_argument('--DA', action='store_true', help='use Color Data Augmentation')
parser.add_argument('--resume', action='store_true', help='use resume trainning')
parser.add_argument('--share', action='store_true', help='share weight between different view')
parser.add_argument('--extra_Google', action='store_true', help='using extra noise Google')
parser.add_argument('--LPN', action='store_true', help='use LPN')
parser.add_argument('--decouple', action='store_true', help='use decouple')
parser.add_argument('--block', default=4, type=int, help='the num of block')
parser.add_argument('--scale', default=1 / 32, type=float, metavar='S', help='scale the loss')
parser.add_argument('--lambd', default=3.9e-3, type=float, metavar='L', help='weight on off-diagonal terms')
parser.add_argument('--g', default=0.9, type=float, metavar='L', help='weight on loss and deloss')
parser.add_argument('--t', default=4.0, type=float, metavar='L', help='temperature of conv matrix')
parser.add_argument('--experiment_name', default='debug', type=str, help='log dir name')
parser.add_argument('--adam', action='store_true', help='using adam optimization')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--balance', action='store_true', help='using balance sampler')
parser.add_argument('--select_id', action='store_true', help='select id')
parser.add_argument('--multi_image', action='store_true', help='only inputs3 + inputs3_s training')
parser.add_argument('--expand_id', action='store_true', help='expand id')
parser.add_argument('--dro_lead', action='store_true', help='drone leading sampling')
parser.add_argument('--sat_lead', action='store_true', help='satellite leading sampling')
parser.add_argument('--normal', action='store_true', help='normal training')
parser.add_argument('--only_decouple', action='store_true', help='do not use balance losss')
parser.add_argument('--e1', default=1, type=int, help='the exponent of on diag')
parser.add_argument('--e2', default=1, type=int, help='the exponent of off diag')
parser.add_argument('--swin', action='store_true', help='using swin as backbone')
parser.add_argument('--fp16', action='store_true',
                    help='use float16 instead of float32, which will save about 50% memory')
parser.add_argument('--test_dir', default='/home/lihaoran/BJDD_datesets/datesets/University-Release/test',
                    help='use float16 instead of float32, which will save about 50% memory')
parser.add_argument('--modelNum', type=int, help='用来区分模型')
parser.add_argument('--val_batchsize', default=128, type=int, help='batchsize')
parser.add_argument('--SAM', type=int, default=0, help='用来区分模型')
parser.add_argument('--infonce', type=int, default=1, help='采用infonce损失来平衡')
parser.add_argument('--Twozerothree', action='store_true', help='采用infonce损失来平衡')
opt = parser.parse_args()

os.environ["WANDB_API_KEY"] = 'c67d9a2bf298e65e8717d5c693270e77d117bddb'
os.environ["WANDB_MODE"] = "offline"
wandb.init(project="DWDR", name=opt.name)


def seed_torch(seed=opt.seed):
    # random.seed(seed)
    seed = 1234
    # print("1111111")
    # os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # random.seed(seed)  # Python random module.
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.use_deterministic_algorithms(False)


print('random seed---------------------:', opt.seed)
seed_torch(opt.seed)

if opt.resume:
    model, opt, start_epoch = load_network(opt.name, opt)
else:
    start_epoch = 0

# debug
# opt.LPN=True
# opt.decouple = True


fp16 = opt.fp16
data_dir = opt.data_dir
test_dir = opt.test_dir
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >= 0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids) > 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    cudnn.enabled = True
    cudnn.benchmark = True
    print('-----------------------------use multi-GPU',gpu_ids)

else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    cudnn.benchmark = True
    testGpu = ''.join(map(str, gpu_ids))
print('---------------Pool Strategy------------:', opt.pool)
######################################################################
# Load Data
# ---------
#

transform_train_list = [
    # transforms.RandomResizedCrop(size=(opt.h, opt.w), scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.Pad(opt.pad, padding_mode='edge'),
    transforms.RandomCrop((opt.h, opt.w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_satellite_list = [
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.Pad(opt.pad, padding_mode='edge'),
    transforms.RandomAffine(90),
    transforms.RandomCrop((opt.h, opt.w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_val_list = [
    transforms.Resize(size=(opt.h, opt.w), interpolation=3),  # Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

if opt.erasing_p > 0:
    transform_train_list = transform_train_list + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                   hue=0)] + transform_train_list
    transform_satellite_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                       hue=0)] + transform_satellite_list

if opt.DA:
    transform_train_list = [ImageNetPolicy()] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
    'satellite': transforms.Compose(transform_satellite_list)
}

transform_move_list = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_all = ''
if opt.train_all:
    train_all = '_all'

image_datasets = {}
if opt.expand_id:
    print('--------------------expand id-----------------------')
    image_datasets['satellite'] = ImageFolder_expandID(os.path.join(data_dir, 'satellite'),
                                                       transform=data_transforms['satellite'])
else:
    image_datasets['satellite'] = SatData(data_dir, data_transforms['satellite'], data_transforms['train'])

if opt.select_id:
    print('--------------------select id-----------------------')
    image_datasets['drone'] = ImageFolder_selectID(os.path.join(data_dir, 'drone'), transform=data_transforms['train'])
else:
    image_datasets['drone'] = DroneData(data_dir, data_transforms['train'], data_transforms['satellite'])


def _init_fn(worker_id):
    np.random.seed(int(opt.seed) + worker_id)


dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=True, num_workers=8, pin_memory=False, worker_init_fn=_init_fn)
               # 8 workers may work faster
               for x in ['satellite', 'drone']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['satellite', 'drone']}
class_names = image_datasets['satellite'].classes
print(dataset_sizes)

test_imgDatasets = {x: datasets.ImageFolder(os.path.join(test_dir, x), data_transforms['val']) for x in
                    ['gallery_satellite', 'gallery_drone']}
for x in ['query_satellite', 'query_drone']:
    test_imgDatasets[x] = customData(os.path.join(test_dir, x), data_transforms['val'], rotate=0)
Val_dataloaders = {x: torch.utils.data.DataLoader(test_imgDatasets[x], batch_size=opt.val_batchsize,
                                                  shuffle=False, num_workers=8) for x in
                   ['gallery_satellite', 'gallery_drone', 'query_satellite', 'query_drone']}

use_gpu = torch.cuda.is_available()

######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

y_loss = {}  # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


def evaluate(qf, ql, gf, gl):
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)
    good_index = query_index
    # print(good_index)
    # print(index[0:10])
    junk_index = np.argwhere(gl == -1)

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def which_view(name):
    if 'satellite' in name:
        return 1
    elif 'street' in name:
        return 2
    elif 'drone' in name:
        return 3
    else:
        print('unknown view')
    return -1


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def get_id(img_path):
    camera_id = []
    labels = []
    paths = []
    for path, v in img_path:
        # print(path, v)
        folder_name = os.path.basename(os.path.dirname(path))
        labels.append(int(folder_name))
        paths.append(path)
    return labels, paths


def extract_feature(model, dataloaders, view_index=1):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        # print(count)
        ff = torch.FloatTensor(n, 512).zero_().cuda()
        # if opt.swin:
        #     ff = torch.FloatTensor(n,1024).zero_().cuda()
        if opt.LPN:
            # ff = torch.FloatTensor(n,2048,6).zero_().cuda()
            ff = torch.FloatTensor(n, 512, opt.block).zero_().cuda()

        for i in range(2):
            if (i == 1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs = model(input_img)
            # print(outputs.shape)
            if opt.decouple:
                ff += outputs[0]
            elif opt.infonce == 1:
                ff += outputs[0]
            else:
                ff += outputs
        # norm feature
        if opt.LPN:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(opt.block)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features, ff.data.cpu()), 0)
    return features


# work channel loss
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def decouple_loss(y1, y2, scale_loss, lambd):
    batch_size = y1.size(0)
    c = y1.T @ y2
    c.div_(batch_size)
    on_diag = torch.diagonal(c)
    p_on = (1 - on_diag) / 2
    on_diag = torch.pow(p_on, opt.e1) * torch.pow(torch.diagonal(c).add_(-1), 2)
    on_diag = on_diag.sum().mul(scale_loss)

    off_diag = off_diagonal(c)
    p_off = torch.abs(off_diag)
    off_diag = torch.pow(p_off, opt.e2) * torch.pow(off_diagonal(c), 2)
    off_diag = off_diag.sum().mul(scale_loss)
    loss = on_diag + off_diag * lambd
    return loss, on_diag, off_diag * lambd


def one_LPN_output(outputs, labels, criterion, block):
    # part = {}
    sm = nn.Softmax(dim=1)
    num_part = block
    score = 0
    loss = 0
    for i in range(num_part):
        part = outputs[i]
        score += sm(part)
        loss += criterion(part, labels)

    _, preds = torch.max(score.data, 1)

    return preds, loss


def one_info_output(Soutputs, Doutputs, labels, criterion, block):
    num_part = block
    for i in range(num_part):
        Dpart = Soutputs[i]  # 2,701
        Spart = Doutputs[i]  # 2,701
        s_norm = F.normalize(Dpart, dim=1)
        d_norm = F.normalize(Spart, dim=1)
        features = torch.cat([s_norm, d_norm], dim=1)  # 2,701


def val(model, epoch, ):
    config_path = os.path.join('./model', opt.name, 'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    model_test = copy.deepcopy(model)

    if opt.LPN:
        if len(gpu_ids) > 1:
            for i in range(opt.block):
                cls_name = 'classifier' + str(i)
                c = getattr(model_test.module, cls_name)
                c.classifier = nn.Sequential()
        # model = three_view_net_test(model)

        else:
            for i in range(opt.block):
                cls_name = 'classifier' + str(i)
                c = getattr(model_test, cls_name)
                c.classifier = nn.Sequential()
    else:
        model_test.classifier.classifier = nn.Sequential()
        # model.classifier = nn.Sequential()
    model_test = model_test.eval()
    model_test = model_test.cuda()
    query_name = 'query_drone'
    gallery_name = 'gallery_satellite'
    which_gallery = which_view(gallery_name)
    which_query = which_view(query_name)
    print('%d -> %d:' % (which_query, which_gallery))
    gallery_path = test_imgDatasets[gallery_name].imgs
    f = open('gallery_name.txt', 'w')
    for p in gallery_path:
        f.write(p[0] + '\n')
    query_path = test_imgDatasets[query_name].imgs
    f = open('query_name.txt', 'w')
    for p in query_path:
        f.write(p[0] + '\n')
    gallery_label, gallery_path = get_id(gallery_path)
    query_label, query_path = get_id(query_path)

    with torch.no_grad():
        query_feature = extract_feature(model_test, Val_dataloaders[query_name], which_query)
        gallery_feature = extract_feature(model_test, Val_dataloaders[gallery_name], which_gallery)

        time_elapsed = time.time() - since
        print('Test complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        # Save to Matlab for check
        result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_path': gallery_path,
                  'query_f': query_feature.numpy(), 'query_label': query_label, 'query_path': query_path}
        scipy.io.savemat('pytorch_result.mat', result)

        print(opt.name)
        result = './model/%s/result.txt' % opt.name
        # os.system(
        #     'CUDA_VISIBLE_DEVICES=%d python evaluate_gpu.py --name %s | tee -a %s' % (testGpu, opt.name, result))

        result = scipy.io.loadmat('pytorch_result.mat')
        query_feature = torch.FloatTensor(result['query_f'])
        query_label = result['query_label'][0]
        gallery_feature = torch.FloatTensor(result['gallery_f'])
        gallery_label = result['gallery_label'][0]
        multi = os.path.isfile('multi_query.mat')

        if multi:
            m_result = scipy.io.loadmat('multi_query.mat')
            mquery_feature = torch.FloatTensor(m_result['mquery_f'])
            mquery_label = m_result['mquery_label'][0]
            mquery_feature = mquery_feature.cuda()

        query_feature = query_feature.cuda()
        gallery_feature = gallery_feature.cuda()

        CMC = torch.IntTensor(len(gallery_label)).zero_()
        ap = 0.0

        for i in range(len(query_label)):
            ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], gallery_feature, gallery_label)
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp

        CMC = CMC.float()
        CMC = CMC / len(query_label)  # average CMC
        print(round(len(gallery_label) * 0.01))
        acc1 = CMC[0] * 100
        ap1 = ap / len(query_label) * 100
        print('Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f' % (
            acc1, CMC[4] * 100, CMC[9] * 100, CMC[round(len(gallery_label) * 0.01)] * 100, ap1))

        # 放到wandb中去

        # multiple-query
        CMC = torch.IntTensor(len(gallery_label)).zero_()
        ap = 0.0
        if multi:
            for i in range(len(query_label)):
                mquery_index1 = np.argwhere(mquery_label == query_label[i])
                mquery_index2 = np.argwhere(mquery_cam == query_cam[i])
                mquery_index = np.intersect1d(mquery_index1, mquery_index2)
                mq = torch.mean(mquery_feature[mquery_index, :], dim=0)
                ap_tmp, CMC_tmp = evaluate(mq, query_label[i], query_cam[i], gallery_feature, gallery_label,
                                           gallery_cam)
                if CMC_tmp[0] == -1:
                    continue
                CMC = CMC + CMC_tmp
                ap += ap_tmp
                # print(i, CMC_tmp[0])
            CMC = CMC.float()
            CMC = CMC / len(query_label)  # average CMC
            print('multi Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))

        return acc1, ap1


def train_model(model, model_test, criterion, optimizer, scheduler, epoch, warm_up, warm_iteration, num_epochs=25):
    epoch = epoch + start_epoch
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train']:
        if phase == 'train':
            model.train(True)  # Set model to training mode
        else:
            model.train(False)  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0.0
        running_corrects3 = 0.0
        ins_loss = 0.0
        dec_loss = 0.0
        on_loss = 0.0
        off_loss = 0.0
        lossinfo1 = 0.0
        lossinfo2 = 0.0
        # Iterate over data.
        for data, data3 in zip(dataloaders['satellite'], dataloaders['drone']):
            # get the inputs
            inputs, inputs_d, labels = data
            # print(inputs.shape,"inputs")  # torch.Size([8, 3, 224, 224])
            # print(inputs_d.shape,"inputs_d")  # torch.Size([8, 3, 224, 224])
            # print(labels,"labels")  # tensor([360, 142, 532, 398, 322, 141, 199, 609])
            inputs3, inputs3_s, labels3 = data3
            now_batch_size, c, h, w = inputs.shape
            if now_batch_size < opt.batchsize:  # skip the last batch
                continue
            if use_gpu:
                if opt.normal:
                    inputs = Variable(inputs.cuda().detach())
                    inputs3 = Variable(inputs3.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                    labels3 = Variable(labels3.cuda().detach())
                else:
                    inputs = Variable(inputs.cuda().detach())
                    inputs_d = Variable(inputs_d.cuda().detach())
                    inputs3 = Variable(inputs3.cuda().detach())
                    inputs3_s = Variable(inputs3_s.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                    labels3 = Variable(labels3.cuda().detach())

            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            if opt.decouple:
                if opt.infonce == 1:
                    outs_c, outs_f, outs_info = model(inputs)
                else:
                    outs_c, outs_f = model(inputs)
            else:
                if opt.infonce == 1:
                    outs_c, outs_info = model(inputs)
                else:
                    outs_c = model(inputs)
            if opt.balance:
                if opt.decouple:
                    if opt.infonce == 1:
                        outs_d_c, outs_d_f, outs_d_info = model(inputs_d)
                    else:
                        outs_d_c, outs_d_f = model(inputs_d)
                else:
                    if opt.infonce == 1:
                        outs_d_c, outs_d_info = model(inputs_d)
                    else:
                        outs_d_c = model(inputs_d)

            if opt.decouple:
                if opt.infonce == 1:
                    outd_c, outs3_f, outd_info = model(inputs3)
                else:
                    outd_c, outs3_f = model(inputs3)
            else:
                if opt.infonce == 1:
                    outd_c, outd_info = model(inputs3)
                else:
                    outd_c = model(inputs3)
            if opt.balance:
                if opt.decouple:
                    if opt.infonce == 1:
                        outs3_s_c, outs3_s_f, outs3_s_info = model(inputs3_s)
                    else:
                        outs3_s_c, outs3_s_f = model(inputs3_s)
                else:
                    if opt.infonce == 1:
                        outs3_s_c, outs3_s_info = model(inputs3_s)
                    else:
                        outs3_s_c = model(inputs3_s)
            # calculate loss
            if opt.LPN:
                if opt.balance:
                    # print('--------------------- using data balance---------------------------')
                    if opt.only_decouple:
                        print('--------------------- only decouple---------------------------')
                        preds, loss = one_LPN_output(outs_c, labels, criterion, opt.block)
                        preds3, loss3 = one_LPN_output(outd_c, labels3, criterion, opt.block)
                        loss = loss + loss3
                    else:

                        preds, loss = one_LPN_output(outs_c, labels, criterion, opt.block)
                        _, loss_d = one_LPN_output(outs_d_c, labels, criterion, opt.block)
                        loss = loss + loss_d
                        preds3, loss3 = one_LPN_output(outd_c, labels3, criterion, opt.block)
                        _, loss3_s = one_LPN_output(outs3_s_c, labels3, criterion, opt.block)
                        loss3 = loss3 + loss3_s
                        loss = (loss + loss3) / 2

                else:

                    preds, loss = one_LPN_output(outs_c, labels, criterion, opt.block)
                    preds3, loss3 = one_LPN_output(outd_c, labels3, criterion, opt.block)
                    loss = loss + loss3

                if opt.decouple:

                    if opt.balance:
                        deloss1, on, off = decouple_loss(outs_f, outs_d_f, opt.scale, opt.lambd)
                        deloss2, on1, off1 = decouple_loss(outs3_s_f, outs3_f, opt.scale, opt.lambd)
                        deloss = (deloss1 + deloss2) / 2
                        # deloss = deloss2
                        on = (on + on1) / 2
                        off = (off + off1) / 2
                        insloss = loss
                        loss = opt.g * insloss + (1 - opt.g) * deloss

                # 在这里需要加入infonce的loss计算方法  在使用balance的时候需要进行双向的加权（一个正常的loss一个color变化的loss）

                if opt.infonce == 1 and opt.balance :  # 默认使用
                    # 正常图片下的infonce
                    sate = F.normalize(outs_info, dim=1)
                    drone = F.normalize(outd_info, dim=1)
                    sate_ = F.normalize(outs_d_info, dim=1)
                    drone_ = F.normalize(outs3_s_info, dim=1)

                    features1 = torch.cat([sate.unsqueeze(1), sate_.unsqueeze(1)], dim=1)
                    features2 = torch.cat([drone.unsqueeze(1), drone_.unsqueeze(1)], dim=1)

                    loss_info = infonce(features1, labels)
                    loss = loss + loss_info

                    loss_info1 = infonce(features2, labels3)
                    loss = loss + loss_info1
            else:
                pass

            if epoch < opt.warm_epoch and phase == 'train':
                warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                loss *= warm_up

            if phase == 'train':
                if fp16:  # we use optimier to backward loss
                    loss.backward()
                else:
                    loss.backward()
                if opt.SAM != 1:
                    optimizer.step()
                else:
                    optimizer.first_step(zero_grad=True)
                    # sam中计算第二次梯度
                    if opt.balance:

                        if opt.decouple:  # 同时使用balance和decouple的情况下

                            if opt.infonce == 1:
                                outs_c, outs_f, outs_info = model(inputs)
                                outd_c, outs3_f, outd_info = model(inputs3)
                                outs_d_c, outs_d_f, outs_d_info = model(inputs_d)
                                outs3_s_c, outs3_s_f, outs3_s_info = model(inputs3_s)

                            else:
                                outs_c, outs_f = model(inputs)
                                outd_c, outs3_f = model(inputs3)
                                outs3_s_c, outs3_s_f = model(inputs3_s)
                                outs_d_c, outs_d_f = model(inputs_d)

                            preds, loss = one_LPN_output(outs_c, labels, criterion, opt.block)
                            _, loss_d = one_LPN_output(outs_d_c, labels, criterion, opt.block)
                            loss = loss + loss_d
                            preds3, loss3 = one_LPN_output(outd_c, labels3, criterion, opt.block)
                            _, loss3_s = one_LPN_output(outs3_s_c, labels3, criterion, opt.block)
                            loss3 = loss3 + loss3_s
                            loss = (loss + loss3) / 2
                            deloss1, on, off = decouple_loss(outs_f, outs_d_f, opt.scale, opt.lambd)
                            deloss2, on1, off1 = decouple_loss(outs3_s_f, outs3_f, opt.scale, opt.lambd)
                            deloss = (deloss1 + deloss2) / 2
                            # deloss = deloss2
                            on = (on + on1) / 2
                            off = (off + off1) / 2
                            insloss = loss
                            loss = opt.g * insloss + (1 - opt.g) * deloss

                        else:

                            if opt.infonce == 1:

                                outd_c, outd_info = model(inputs3)
                                outs_c, outs_info = model(inputs)
                                outs_d_c, outs_d_info = model(inputs_d)
                                outs3_s_c, outs3_s_info = model(inputs3_s)

                            else:

                                outs_c = model(inputs)
                                outd_c = model(inputs3)
                                outs3_s_c = model(inputs3_s)
                                outs_d_c = model(inputs_d)

                            preds, loss = one_LPN_output(outs_c, labels, criterion, opt.block)
                            _, loss_d = one_LPN_output(outs_d_c, labels, criterion, opt.block)
                            loss = loss + loss_d
                            preds3, loss3 = one_LPN_output(outd_c, labels3, criterion, opt.block)
                            _, loss3_s = one_LPN_output(outs3_s_c, labels3, criterion, opt.block)
                            loss3 = loss3 + loss3_s
                            loss = (loss + loss3) / 2

                        # 使用infonce的情况下计算新一轮的损失？


                    else:

                        # 在不适用balance和decouple的时候去使用infonce

                        if opt.infonce == 1:
                            outs_c, outs_info = model(inputs)
                            outd_c, outd_info = model(inputs3)
                        else:
                            outs_c = model(inputs)
                            outd_c = model(inputs3)

                        preds, lossmin3 = one_LPN_output(outs_c, labels, criterion, opt.block)
                        preds3, lossmin4 = one_LPN_output(outd_c, labels3, criterion, opt.block)
                        loss = (lossmin3 + lossmin4)

                    if opt.infonce == 1 and opt.balance:
                        # 最后统一计算损失
                        sate = F.normalize(outs_info, dim=1)
                        drone = F.normalize(outd_info, dim=1)
                        sate_ = F.normalize(outs_d_info, dim=1)
                        drone_ = F.normalize(outs3_s_info, dim=1)

                        features1 = torch.cat([sate.unsqueeze(1), sate_.unsqueeze(1)], dim=1)
                        features2 = torch.cat([drone.unsqueeze(1), drone_.unsqueeze(1)], dim=1)

                        loss_info = infonce(features1, labels)
                        loss = loss + loss_info

                        loss_info1 = infonce(features2, labels3)
                        loss = loss + loss_info1

                    if epoch < opt.warm_epoch and phase == 'train':
                        loss *= warm_up
                    # print(loss2, "2")
                    if fp16:  # we use optimier to backward loss
                        loss.backward()
                    else:
                        loss.backward()
                    optimizer.second_step(zero_grad=True)
                ##########
            if opt.moving_avg < 1.0:
                update_average(model_test, model, opt.moving_avg)

            # statistics

            running_loss += loss.item() * now_batch_size
            if opt.decouple:
                ins_loss += insloss.item() * now_batch_size
                dec_loss += deloss.item() * now_batch_size
                on_loss += on.item() * now_batch_size
                off_loss += off.item() * now_batch_size

            running_corrects += float(torch.sum(preds == labels.data))
            running_corrects3 += float(torch.sum(preds3 == labels3.data))

        epoch_loss = running_loss / dataset_sizes['satellite']
        epoch_acc = running_corrects / dataset_sizes['satellite']
        epoch_acc3 = running_corrects3 / dataset_sizes['satellite']

        if opt.decouple:
            epoch_ins_loss = ins_loss / dataset_sizes['satellite']
            epoch_dec_loss = dec_loss / dataset_sizes['satellite']
            epoch_on_loss = on_loss / dataset_sizes['satellite']
            epoch_off_loss = off_loss / dataset_sizes['satellite']

        if opt.infonce == 1:
            lossinfo1 += loss_info.item() * now_batch_size
            # lossinfo2 += loss_info1.item() * now_batch_size
            epoch_loss_info1 = lossinfo1 / dataset_sizes['satellite']
            # epoch_loss_info2 = lossinfo2 / dataset_sizes['satellite']

        if opt.decouple:
            print(
                '{} Loss: {:.4f} Satellite_Acc: {:.4f} Drone_Acc: {:.4f}, On_Loss: {:.4f}, Off_Loss: {:.4f},'.format(
                    phase, epoch_loss, epoch_acc, epoch_acc3, epoch_on_loss, epoch_off_loss))

        if opt.infonce == 1:
            print('{} Loss: {:.4f} Satellite_Acc: {:.4f} Drone_Acc: {:.4f} infoloss1: {:.4f} infoloss2: {:.4f}'.format(
                phase, epoch_loss, epoch_acc,
                epoch_acc3, epoch_loss_info1, 0.00))

        else:
            print('{} Loss: {:.4f} Satellite_Acc: {:.4f} Drone_Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc,
                                                                                   epoch_acc3))

        y_loss[phase].append(epoch_loss)
        y_err[phase].append(1.0 - epoch_acc)

        # saving last model:
        if phase == 'train':
            scheduler.step()
        if epoch + 1 == num_epochs and len(gpu_ids) > 1:
            save_network(model.module, opt.name, epoch)
        elif epoch + 1 > 100 and (epoch + 1) % 10 == 0:
            save_network(model, opt.name, epoch)
        # draw_curve(epoch)

    if epoch % 4 == 1:
        wandb.log({
            'Step': epoch + 1,
            'Loss': epoch_loss,
            'Satellite_Acc': epoch_acc,
            'Drone_Acc': epoch_acc3,
        })

    return model, warm_up


######################################################################
# Draw Curve
# ---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(os.path.join('./model', name, 'train.jpg'))


######################################################################
# Finetuning the convnet
# ----------------------

# Load a pretrainied model and reset final fully connected layer.
#
if opt.LPN:
    # model = ft_net_LPN(len(class_names), droprate=opt.droprate, stride=opt.stride, pool=opt.pool, block=opt.block,
    #                    decouple=opt.decouple)
    model = CSWinTransv2_threeIn(len(class_names), droprate=opt.droprate, decouple=opt.decouple, infonce=opt.infonce)

    if opt.Twozerothree:
        from model_203 import *

        model = CSWinTrans_attention(len(class_names), droprate=opt.droprate, decouple=opt.decouple,
                                     infonce=opt.infonce)

    # model = CSWinTrans_twoStage(len(class_names), droprate=opt.droprate)
# elif opt.swin:
#     model = ft_net_swin(len(class_names), droprate=opt.droprate, decouple=opt.decouple)
# else:
#     model = mainModule(len(class_names), droprate=opt.droprate, decouple=opt.decouple)
#     # model = TransCnn_50(len(class_names), droprate=opt.droprate)
#     # model = SAIG_Shallow(img_size=224)

# model = two_view_net()

opt.nclasses = len(class_names)
print('nclass--------------------:', opt.nclasses)
# print(model)
# For resume:
if start_epoch >= 40:
    opt.lr = opt.lr * 0.1

if not opt.LPN:
    model = model.cuda()

    params = [{"params": model.get_1x_lr_params(), "lr": opt.lr * 0.1},
              {"params": model.get_10x_lr_params(), "lr": opt.lr},
              # {"params": model.model.get_add_lr_params(), "lr": opt.lr * 1.5 }
              ]

    # ignored_params = list(map(id, model.classifier.parameters()))
    # # ignored_params.append(map(id,model.model.t2tswin.parameters()))
    # base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD(
        params,
        #     [
        #     {'params': base_params, 'lr': 0.1 * opt.lr},
        #     {'params': model.classifier.parameters(), 'lr': opt.lr},
        #     # {'params':model.model.t2tswin.parameters(),'lr':opt.lr}
        # ],
        weight_decay=5e-4, momentum=0.9, nesterov=True)

    # print(ignored_params)

    # print(base_params,"base_params")
    # print(model.model.classifier,"model.model.classifier.parameters()")

else:
    # ignored_params = list(map(id, model.model.fc.parameters() ))
    if len(gpu_ids) > 1:
        print(gpu_ids,"gpu_idsgpu_idsgpu_idsgpu_idsgpu_idsgpu_idsgpu_idsgpu_idsgpu_idsgpu_idsgpu_ids")
        model = torch.nn.DataParallel(model,device_ids=[0,1]).cuda()

        ignored_params = list()
        for i in range(opt.block):
            cls_name = 'classifier' + str(i)
            c = getattr(model.module, cls_name)
            ignored_params += list(map(id, c.parameters()))

        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

        optim_params = [{'params': base_params, 'lr': 0.1 * opt.lr}]
        for i in range(opt.block):
            cls_name = 'classifier' + str(i)
            c = getattr(model.module, cls_name)
            optim_params.append({'params': c.parameters(), 'lr': opt.lr})

        # optim_params = [{"params": model.module.get_1x_lr_params(), "lr": opt.lr * 0.1},
        #                 {"params": model.module.get_10x_lr_params(), "lr": opt.lr},
        #                 # {"params": model.model.get_add_lr_params(), "lr": opt.lr * 1.5 }
        #                 ]

    else:

        model = model.cuda()

        print('---------------------use one gpu-----------------------')
        ignored_params = list()
        # ignored_params += list(map(id, model.rdim.parameters() ))
        for i in range(opt.block):
            cls_name = 'classifier' + str(i)
            c = getattr(model, cls_name)
            ignored_params += list(map(id, c.parameters()))

        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

        optim_params = [{'params': base_params, 'lr': 0.1 * opt.lr}]
        # optim_params.append({'params': model.rdim.parameters(), 'lr': opt.lr})
        for i in range(opt.block):
            cls_name = 'classifier' + str(i)
            c = getattr(model, cls_name)
            optim_params.append({'params': c.parameters(), 'lr': opt.lr})

        # 学习率调整需要更改

        # optim_params = [{"params": model.get_1x_lr_params(), "lr": opt.lr * 0.1},
        #                 {"params": model.get_10x_lr_params(), "lr": opt.lr},
        #                 # {"params": model.model.get_add_lr_params(), "lr": opt.lr * 1.5 }
        #             ]

    # optimizer_ft = optim.SGD(optim_params, weight_decay=5e-4, momentum=0.9, nesterov=True)
    # base_optimizer = optim.SGD(optim_params, weight_decay=5e-4, momentum=0.9, nesterov=True)
    # base_optimizer = torch.optim.AdamW
    # optimizer_ft = SAM(optim_params, base_optimizer, lr=opt.lr, betas=(0.9, 0.999), weight_decay=5e-4, amsgrad=False,
    #                    adaptive=True, rho=2.5)

    infonce = SupConLoss(temperature=0.1)

    if opt.adam:
        optimizer_ft = optim.Adam(optim_params, opt.lr, weight_decay=5e-4)

    if opt.SAM == 1:
        base_optimizer = torch.optim.SGD
        optimizer_ft = SAM(optim_params, base_optimizer, lr=opt.lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
        # base_optimizer = torch.optim.Adam
        # optimizer_ft = SAM(optim_params, base_optimizer, lr=opt.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4, )
    else:
        optimizer_ft = optim.SGD(optim_params, weight_decay=5e-4, momentum=0.9, nesterov=True)

# Decay LR by a factor of 0.1 every 40 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=80, gamma=0.1)
exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[120, 180, 210], gamma=0.1)

# exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[60,120,160], gamma=0.1)
# exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=120, eta_min=0.001)
######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU.
#
# neptune.init('wtyu/decouple')
# neptune.create_experiment('LPN+norm(batch*512*4)')


if fp16:
    model, optimizer_ft = amp.initialize(model, optimizer_ft, opt_level="O1")

# if len(gpu_ids)>1:
#     model = torch.nn.DataParallel(model, [3,2]).cuda()
# else:
#     model = model.cuda()

criterion = nn.CrossEntropyLoss()

if opt.moving_avg < 1.0:
    model_test = copy.deepcopy(model)
    # num_epochs = 140
    num_epochs = 210
else:
    model_test = None
    # num_epochs = 120
    num_epochs = 210

warm_up = 0.1  # We start from the 0.1*lrRate
bestAcc = 0
bestAp = 0
bestEp = 0
since = time.time()

warm_iteration = round(dataset_sizes['satellite'] / opt.batchsize) * opt.warm_epoch  # first 5 epoch

for epoch in range(num_epochs - start_epoch):

    # train
    model, warm_up = train_model(model, model_test, criterion, optimizer_ft, exp_lr_scheduler, epoch, warm_up,
                                 warm_iteration,
                                 num_epochs=num_epochs)
    warm_up = warm_up

    dir_name = os.path.join('./model', name)
    if not opt.resume:
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        # record every run
        copyfile('./run_mul_gpu_view.sh', dir_name + '/run_mul_gpu_view.sh')
        copyfile('./train_info.py', dir_name + '/train_info.py')
        copyfile('./model.py', dir_name + '/model.py')
        # save opts
        with open('%s/opts.yaml' % dir_name, 'w') as fp:
            yaml.dump(vars(opt), fp, default_flow_style=False)

    # val
    # if epoch % 1 == 0:
    if epoch > 100 and epoch % 10 == 0:
        save_network(model, opt.name, epoch)
        acc1, ap1 = val(model, epoch)
        wandb.log({
            'acc@1': acc1,
            'ap': ap1,
            'Step': epoch + 1
        })
        if acc1 > bestAcc:
            bestAcc = acc1
            bestEp = epoch
        if ap1 > bestAp:
            bestAp = ap1

        print('BestRecall@1:%.2f, BestAP:%.2f,bestepoch:%.0f' % (bestAcc, bestAp, bestEp))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('BestRecall@1:%.2f, BestAP:%.2f,bestepoch:%.0f' % (bestAcc, bestAp, bestEp))
    print("name:", opt.name)
    # if epoch_loss < best_loss:
    #     best_loss = epoch_loss
    #     best_epoch = epoch
    #     last_model_wts = model.state_dict()

    # time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))
    # model.load_state_dict(last_model_wts)
    # if len(gpu_ids)>1:
    #     save_network(model.module, opt.name, 'last')
    #     print('best_epoch:', best_epoch)
    # else:
    #     save_network(model, opt.name, 'last')
    #     print('best_epoch:', best_epoch)

# model = train_model(model, model_test, criterion, optimizer_ft, exp_lr_scheduler,
#                     num_epochs=num_epochs)
# neptune.stop()
print('BestRecall@1:%.2f, BestAP:%.2f' % (bestAcc, bestAp))
save_network(model, "best", epoch)
