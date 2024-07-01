import os
import json
import random
import operator
import functools
import itertools
import multiprocessing

import torch
import torchvision
import numpy as np
import skimage
import pycocotools.mask

import sys
sys.path.append("../..")
from vsrd_toy.operations.geometric_operations import expand_to_4x4


class KITTI360Dataset(torch.utils.data.Dataset):
    '''
    Inputs:
    
    filanems:  ['/media/zliu/data12/dataset/KITTI/VSRD_Format/filenames/R50-N16-M128-B16/2013_05_28_drive_0000_sync/sampled_image_filenames.txt']
    class_names:  ['car']
    nums_source_frames:  16
    
    Type:
    <class 'vsrd.transforms.geometric_transforms.Resizer'>
    target_transforms :
    [Resizer(), MaskAreaFilter(), MaskRefiner(), MaskAreaFilter(), BoxGenerator(), BoxSizeFilter(), SoftRasterizer()]
    
    source_transforms:
    [Resizer(), MaskAreaFilter(), MaskRefiner(), MaskAreaFilter(), BoxGenerator(), BoxSizeFilter(), SoftRasterizer()]
    
    
    Return:
    {'train': <vsrd.datasets.kitti_360_dataset.KITTI360Dataset object at 0x70b2fc35b1c0>}
    
    '''
    def __init__(
        self,
        filenames,
        class_names,
        num_workers=4,
        num_source_frames=2,
        target_transforms=[],
        source_transforms=[],
        rectification=True,
    ):
        
        super().__init__()

        self.image_filenames = [] # save the image list
        self.image_blacklist = set()
        

        for filename in filenames:
            with open(filename) as file:
                for line in file:
                    # 3 colums: 1 is the instance id 
                    _, target_image_filename, source_relative_indices = line.strip().split(" ")
                    source_relative_indices = list(map(int, source_relative_indices.split(",")))
                    self.image_filenames.append((target_image_filename, source_relative_indices))

        self.filenames = filenames
        self.class_names = class_names
        self.num_workers = num_workers
        self.num_source_frames = num_source_frames
        self.target_transforms = target_transforms
        self.source_transforms = source_transforms
        self.rectification = rectification # default is the True



if __name__=="__main__":
    
    
    dataset = KITTI360Dataset(filenames="")
    
    pass