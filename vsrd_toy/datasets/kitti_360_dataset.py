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

from .. import operations


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
        
    @staticmethod
    def get_root_dirname(image_filename):
        # get the root dirname by calling the os.path.dirname 5 times.
        root_dirname = functools.reduce(lambda x, f: f(x), [os.path.dirname] * 5, image_filename)
        return root_dirname

    @staticmethod
    def get_sequence_dirname(image_filename):
        # get the sequence dirname by calling the os.path.dirname 3 times
        sequence_dirname = functools.reduce(lambda x, f: f(x), [os.path.dirname] * 3, image_filename)
        return sequence_dirname

    @staticmethod
    def get_annotation_filename(image_filename):
        annotation_filename = (
            image_filename
            .replace("data_2d_raw", "annotations")
            .replace(".png", ".json")
        )
        return annotation_filename

    @staticmethod
    def get_image_filename(image_filename, relative_index=0):
        frame_index = int(os.path.splitext(os.path.basename(image_filename))[0])
        image_filename = os.path.join(
            os.path.dirname(image_filename),
            f"{frame_index + relative_index:010}.png",
        )
        return image_filename

    @staticmethod
    def read_image(image_filename):
        image = skimage.io.imread(image_filename)
        image = torchvision.transforms.functional.to_tensor(image)
        return image

    def read_annotation(self, annotation_filename):

        with open(annotation_filename) as file:
            annotation = json.load(file)

        intrinsic_matrix = torch.as_tensor(annotation["intrinsic_matrix"])
        extrinsic_matrix = torch.as_tensor(annotation["extrinsic_matrix"])

        instance_ids = {
            class_name: list(masks.keys())
            for class_name, masks in annotation["masks"].items()
            if class_name in self.class_names
        }

        if instance_ids:

            masks = torch.cat([
                torch.as_tensor(np.stack([
                    pycocotools.mask.decode(annotation["masks"][class_name][instance_id])
                    for instance_id in instance_ids
                ]), dtype=torch.float)
                for class_name, instance_ids in instance_ids.items()
            ], dim=0) #(N,H,W)

            labels = torch.cat([
                torch.as_tensor([self.class_names.index(class_name)] * len(instance_ids), dtype=torch.long)
                for class_name, instance_ids in instance_ids.items()
            ], dim=0) # class

            boxes_3d = torch.cat([
                torch.as_tensor([
                    annotation["boxes_3d"][class_name].get(instance_id, [[np.nan] * 3] * 8)
                    for instance_id in instance_ids
                ], dtype=torch.float)
                for class_name, instance_ids in instance_ids.items()
            ], dim=0)

            instance_ids = torch.cat([
                torch.as_tensor(list(map(int, instance_ids)), dtype=torch.long)
                for instance_ids in instance_ids.values()
            ], dim=0)

            return dict(
                masks=masks,
                labels=labels,
                boxes_3d=boxes_3d,
                instance_ids=instance_ids,
                intrinsic_matrix=intrinsic_matrix,
                extrinsic_matrix=extrinsic_matrix,
            )

        else:

            return dict(
                intrinsic_matrix=intrinsic_matrix,
                extrinsic_matrix=extrinsic_matrix,
            )

    def __len__(self):
        return len(self.image_filenames)

    def getitem(self, image_filename, transforms=[]):

        annotation_filename = __class__.get_annotation_filename(image_filename)

        
        # to_tensor 函数将图像的像素值从范围 [0, 255] 缩放到范围 [0.0, 1.0]，
        # 并将图像数据从 HWC (高度, 宽度, 通道) 格式转换为 CHW (通道, 高度, 宽度) 格式，这对于大多数深度学习框架来说是标准格式。
        image = __class__.read_image(image_filename) # get the images
        
        # get annotations
        annotation = self.read_annotation(annotation_filename)
        
        # annotation.setdefault：字典方法，用于获取指定键的值，如果键不存在则插入键并设置为指定值。
        # 创建包含四个键值对的字典，每个值都是空的 PyTorch 张量。
        # 使用 itertools.starmap 将 annotation.setdefault 函数应用于字典的每个键值对。
        # 如果 annotation 字典中不存在某个键，则插入该键，并将值设置为空张量。
        list(itertools.starmap(
            annotation.setdefault,
            dict(
                masks=torch.empty(0, *image.shape[-2:], dtype=torch.float),
                labels=torch.empty(0, dtype=torch.long),
                boxes_3d=torch.empty(0, 8, 3, dtype=torch.float),
                instance_ids=torch.empty(0, dtype=torch.long),
            ).items(),
        ))

        # input shape is annotaions,
        # images
        # image filename
        inputs = dict(
            annotation,
            image=image,
            filename=image_filename,
        )

        for transform in transforms:
            # all the transform
            # including
            # (1) Resizer()
            # (2) MaskAreaFilter()
            # (3) MaskRefiner()
            # (4) MaskAreaFilter()
            # (5) BoxGenerator()
            # (6) BoxSizeFilter()
            # (7) SoftRasterizer()
            inputs = transform(**inputs)

        return inputs

    def __getitem__(self, index):
        
        # get the target filanames/ source relative idxs
        target_image_filename, source_relative_indices = self.image_filenames[index]
        
        
        
        # random return a instance
        if target_image_filename in self.image_blacklist:
            return random.choice(self)

        # get the target inputs
        target_inputs = self.getitem(
            image_filename=target_image_filename,
            transforms=self.target_transforms,
        )
        
        # if no mask: skip
        if not len(target_inputs["masks"]):
            print(f"[{target_image_filename}] No instances. Added to the blacklist.")
            self.image_blacklist.add(target_image_filename)
            return random.choice(self)

        # define a multi inputs: 
        # 0 is the target inputs
        multi_inputs = {0: target_inputs}

        # split it into many small indices, such indices get the med value
        # 仅在子数组 source_relative_indices 非空（即 size 大于 0）时才处理该子数组。
        source_relative_indices = [
            source_relative_indices[len(source_relative_indices) // 2]
            for source_relative_indices
            in np.array_split(source_relative_indices, self.num_source_frames)
            if source_relative_indices.size
        ]

        with multiprocessing.Pool(self.num_workers) as pool:
            multi_inputs.update(dict(zip(
                source_relative_indices,
                pool.imap(
                    functools.partial(self.getitem, transforms=self.source_transforms),
                    [
                        __class__.get_image_filename(
                            image_filename=target_image_filename,
                            relative_index=source_relative_index,
                        )
                        for source_relative_index in source_relative_indices
                    ],
                ),
            )))
            
        # sorted with keys:
        #dict_keys(['extrinsic_matrix', 'filename', 'image', 'intrinsic_matrix', 'labels', 'boxes_3d', 'boxes_2d', 'instance_ids', 'masks', 'hard_masks', 'soft_masks'])
        multi_inputs = dict(sorted(multi_inputs.items(), key=operator.itemgetter(0)))

        if self.rectification:
            
            # target extrinsis matrix--> cam2world.
            target_extrinsic_matrix = target_inputs["extrinsic_matrix"] #[4,4]
            
            # inverse matrix: ----> world2cam.
            #target_extrinsic_matrix[..., :3, :3] 是对这个矩阵进行切片，提取其前 3x3 的子矩阵（旋转部分）
            inverse_target_extrinsic_matrix = torch.linalg.inv(target_extrinsic_matrix)
            
            # 提取外参矩阵的x轴和y轴分量: 从目标外参矩阵中提取 x 轴和 y 轴分量
            x_axis, y_axis, _ = target_extrinsic_matrix[..., :3, :3]
            '''
            x_axis 对应旋转矩阵的第一行。
            y_axis 对应旋转矩阵的第二行。
            _ 对应旋转矩阵的第三行。
            '''
            
            
            # 计算校正角度: 根据校正角度生成旋转矩阵
            rectification_angle = (
                torch.acos(torch.dot(torch.round(y_axis), y_axis)) *
                torch.sign(torch.dot(torch.cross(torch.round(y_axis), y_axis), x_axis))
            )
            # 根据校正角度生成旋转矩阵.
            rectification_matrix = operations.rotation_matrix_x(rectification_angle)

            # 对所有源输入进行校正.: 确保 y_axis 方向对齐。
            # 具体来说，校正旋转矩阵是用来将源相机的 y_axis 对齐到目标相机的 y_axis，并确保旋转的方向（由 x_axis 确定)
            for source_inputs in multi_inputs.values():

                source_extrinsic_matrix = source_inputs["extrinsic_matrix"]
                source_extrinsic_matrix = (
                    source_extrinsic_matrix @
                    inverse_target_extrinsic_matrix @
                    operations.expand_to_4x4(rectification_matrix.T)
                )
                # 更新源输入的外参矩阵和校正矩阵.
                source_inputs.update(
                    extrinsic_matrix=source_extrinsic_matrix,
                    rectification_matrix=rectification_matrix,
                )


        
        # update transforms
        for transforms in [self.target_transforms, self.source_transforms]:
            for transform in transforms:
                if hasattr(transform, "update_params"):
                    transform.update_params()

        return multi_inputs
