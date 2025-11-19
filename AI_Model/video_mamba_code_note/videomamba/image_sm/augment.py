# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
3Augment implementation
Data-augmentation (DA) based on dino DA (https://github.com/facebookresearch/dino)
and timm DA(https://github.com/rwightman/pytorch-image-models)
"""
import torch
from torchvision import transforms

# error: cannot import name '_pil_interp' from 'timm.data.transforms' 
# from timm.data.transforms import _pil_interp, RandomResizedCropAndInterpolation, ToNumpy, ToTensor

# fix: timm version problem
# from timm.data.transforms import str_pil_interp as _pil_interp
from timm.data.transforms import RandomResizedCropAndInterpolation, ToNumpy, ToTensor

import numpy as np
from torchvision import datasets, transforms
import random



from PIL import ImageFilter, ImageOps    # renyu: python图像标准库，也用到了其中ImageFilter的高斯模糊操作，ImageOps的超过阈值反相操作
import torchvision.transforms.functional as TF    # renyu: pytorch图像库，做预处理、变换等操作


# renyu: 高斯模糊操作，参数还是一个范围内的随机值
class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob    # renyu: 按设定概率触发
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(    # renyu: 直接用的PIL.ImageFilter.GaussianBlur方法
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img

# renyu: 像素值翻转（1变254），应该就是反相的意思，不过可以设定一定概率随机反相，那算是噪声了吧
class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:    # renyu: 按设定概率触发
            return ImageOps.solarize(img)    # renyu: 直接用的PIL.ImageOps.solarize方法
        else:
            return img

# renyu: 彩色图像转灰度图，不过还是3通道格式不变
class gray_scale(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p
        self.transf = transforms.Grayscale(3)    # renyu: 直接用的torchvision.transforms.Grayscale方法
 
    def __call__(self, img):
        if random.random() < self.p:    # renyu: 按设定概率触发
            return self.transf(img)
        else:
            return img
 
    
# renyu: 竖直翻转
class horizontal_flip(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2,activate_pred=False):
        self.p = p
        self.transf = transforms.RandomHorizontalFlip(p=1.0)    # renyu: 直接用的torchvision.transforms.RandomHorizontalFlip方法
 
    def __call__(self, img):
        if random.random() < self.p:    # renyu: 按设定概率触发
            return self.transf(img)
        else:
            return img
        
    
# renyu: 整体封装好的数据增强方法，分为三层处理，裁剪-变换-归一化
def new_data_aug_generator(args = None):
    img_size = args.input_size
    remove_random_resized_crop = args.src    # renyu: src参数控制是否做sample random crop简单随机裁剪
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    # renyu: 第一层裁剪，根据参数两种处理方式
    primary_tfl = []
    scale=(0.08, 1.0)
    interpolation='bicubic'
    if remove_random_resized_crop:
        primary_tfl = [
            transforms.Resize(img_size, interpolation=3),
            transforms.RandomCrop(img_size, padding=4,padding_mode='reflect'),
            transforms.RandomHorizontalFlip()
        ]
    else:
        primary_tfl = [
            RandomResizedCropAndInterpolation(
                img_size, scale=scale, interpolation=interpolation),
            transforms.RandomHorizontalFlip()
        ]

        
    # renyu: 第二层变换，随机做灰度、反向、高斯模糊
    secondary_tfl = [transforms.RandomChoice([gray_scale(p=1.0),
                                              Solarization(p=1.0),
                                              GaussianBlur(p=1.0)])]
   
    # renyu: 开启了彩色抖动参数在第二层变换再添加colorjitter
    if args.color_jitter is not None and not args.color_jitter==0:
        secondary_tfl.append(transforms.ColorJitter(args.color_jitter, args.color_jitter, args.color_jitter))
    # renyu: 第三层是归一化
    final_tfl = [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ]
    return transforms.Compose(primary_tfl+secondary_tfl+final_tfl)
