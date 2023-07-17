import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from model import ft_net
from utils import load_network
import yaml
import argparse
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
import cv2 as cv
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
parser = argparse.ArgumentParser(description='Training')

parser.add_argument('--data_dir',default='/home/lihaoran/BJDD_datesets/datesets/University-Release/test',type=str, help='./test_data')
# parser.add_argument('--name', default='two_view_long_share_d0.75_256_s1_lr0.01', type=str, help='save model path')
parser.add_argument('--name', default='lhr_7.5_LPN_sam_nce_201', type=str, help='save model path')
parser.add_argument('--batchsize', default=1, type=int, help='batchsize')
parser.add_argument('--modelNum', type=int,help='用来区分模型')
parser.add_argument('--epoch', default=110 ,type=int,help='训练选择固定的epoch权重')
opt = parser.parse_args()

config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream)
opt.fp16 = config['fp16']
# opt.PCB = config['PCB']
opt.stride = config['stride']
opt.block = config['block']
opt.views = config['views']

if 'h' in config:
    opt.h = config['h']
    opt.w = config['w']
if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else:
    opt.nclasses = 751


def heatmap2d(img, arr):
    # fig = plt.figure()
    # ax0 = fig.add_subplot(121, title="Image")
    # ax1 = fig.add_subplot(122, title="Heatmap")
    # fig, ax = plt.subplots()
    # ax[0].imshow(Image.open(img))
    plt.figure()
    heatmap = plt.imshow(arr, cmap='viridis')
    plt.axis('off')
    # fig.colorbar(heatmap, fraction=0.046, pad=0.04)
    #plt.show()
    plt.savefig('heatmap/image-82_27_lpncat_last', bbox_inches='tight', pad_inches=0)
    # plt.savefig('heatmap_s-base_07lpn7', bbox_inches='tight', pad_inches=0)

data_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# image_datasets = {x: datasets.ImageFolder( os.path.join(opt.data_dir,x) ,data_transforms) for x in ['satellite']}
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
#                                              shuffle=False, num_workers=1) for x in ['satellite']}

# imgpath = image_datasets['satellite'].imgs
# print(imgpath)
# imgname = 'gallery_drone/0721/image-28.jpeg'
# imgname = 'query_satellite/0721/0721.jpg'
# imgname = 'gallery_satellite/0777/0777.jpg'
imgname = 'gallery_drone/0082/image-27.jpeg'
# imgname = 'gallery_satellite/0011/0011.jpg'
# imgname = 'gallery_drone/0011/image-33.jpeg'
imgpath = os.path.join(opt.data_dir,imgname)
img = Image.open(imgpath)
img = data_transforms(img)
img = torch.unsqueeze(img,0)
print(img.shape)
model, _, epoch = load_network(opt.name, opt)

model = model.eval().cuda()

# data = next(iter(dataloaders['satellite']))
# img, label = data
# with torch.no_grad():
#     x = model.model_3.model.conv1(img.cuda())
#     x = model.model_3.model.bn1(x)
#     x = model.model_3.model.relu(x)
#     x = model.model_3.model.maxpool(x)
#     x = model.model_3.model.layer1(x)
#     x = model.model_3.model.layer2(x)
#     x = model.model_3.model.layer3(x)
#     output = model.model_3.model.layer4(x)
with torch.no_grad():

    # print(model)

    # x = model.model.conv1(img.cuda())
    # x = model.model.bn1(x)
    # x = model.model.relu(x)
    # x = model.model.maxpool(x)
    # x = model.model.layer1(x)
    # x = model.model.layer2(x)
    # x = model.model.layer3(x)
    # output = model.model.layer4(x)


    x = model.model.patch_embed(img.cuda())
    for blk in model.model.layers[0].blocks:
        x = blk(x)
    x = model.model.layers[0].downsample(x)
    for blk in model.model.layers[1].blocks:
        x = blk(x)
    x = model.model.layers[1].downsample(x)
    for element, blk in enumerate(model.model.layers[2].blocks):
        x = blk(x)
        if (element + 1) == 14:
            y31 = x
            # y31 = y31.view(y31.size(0), y31.size(1), 16, 16)

        if (element + 1) == 16:
            y32 = x
            # y32 = y32.view(y32.size(0), y32.size(1), 16, 16)
            # y32 = self.usam(y32)

        if (element + 1) == 18:
            y33 = x
            # y33 = x.transpose(2, 1)
            # y33 = y33.view(y33.size(0), y33.size(1), 16, 16)

    # x = model.model.layers[2].downsample(y33)


    # x = model.model.norm(x)
    print(y31.shape,"y31")
    y31 = y31.transpose(2, 1).view(y31.size(0), 512, 16, 16)
    y32 = y32.transpose(2, 1).view(y32.size(0), 512, 16, 16)
    y33 = y33.transpose(2, 1).view(y33.size(0), 512, 16, 16)
    output = torch.cat([y31,y32,y33],dim=1)


    # output = y4.view(y4.size(0), y4.size(1), 16, 16)



print(output.shape)
heatmap = output.squeeze().sum(dim=0).cpu().numpy()
heatmap = cv.resize(heatmap,(512,512))
print(heatmap.shape)
#test_array = np.arange(100 * 100).reshape(100, 100)
# Result is saved tas `heatmap.png`
heatmap2d(imgpath,heatmap)