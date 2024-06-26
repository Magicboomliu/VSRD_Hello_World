import os
import json
import glob
import argparse
import functools
import multiprocessing

import tqdm
import torch
import torchvision
import cv2 as cv
import numpy as np
import pycocotools.mask

import vsrd


# 定义3D盒子的线段索引，用于绘制3D框
LINE_INDICES = [
    [0, 1], [1, 2], [2, 3], [3, 0], # 底部面
    [4, 5], [5, 6], [6, 7], [7, 4], # 顶部面
    [0, 4], [1, 5], [2, 6], [3, 7],  # 垂直线
]


def visualize_annotations(sequence, root_dirname, out_dirname, class_names, frame_rate):
    
    video_writer = None # 初始化视频写入器为None
    # 获取排序后的图像文件名列表
    image_filenames = sorted(glob.glob(os.path.join(root_dirname, "data_2d_raw", sequence, "image_00", "data_rect", "*.png")))
    for image_filename in tqdm.tqdm(image_filenames):
        
        annotation_filename = image_filename.replace("data_2d_raw", "annotations").replace(".png", ".json")
        # 如果注释文件不存在，则跳过该图像
        if not os.path.exists(annotation_filename): continue

        with open(annotation_filename) as file:
            annotation = json.load(file)
                
        # 获取指定类别的实例ID: car and its instances
        instance_ids = {
            class_name: list(masks.keys())
            for class_name, masks in annotation["masks"].items()
            if class_name in class_names
        }


        # 如果没有符合条件的实例ID，则跳过该图像
        if not instance_ids: continue

        # instance masks in every images, [N,H,W], for each N instance is a instance.
        masks = torch.cat([
            torch.as_tensor(np.stack([
                pycocotools.mask.decode(annotation["masks"][class_name][instance_id])
                for instance_id in instance_ids
            ]), dtype=torch.float)
            for class_name, instance_ids in instance_ids.items()
        ], dim=0)
        
    
        # bounding box3d, where the shape is [N,8,3], N is the same instance with the instance mask
        boxes_3d = torch.cat([
            torch.as_tensor([
                annotation["boxes_3d"][class_name].get(instance_id, [[np.nan] * 3] * 8)
                for instance_id in instance_ids
            ], dtype=torch.float)
            for class_name, instance_ids in instance_ids.items()
        ], dim=0)
        
    
        # 获取相机内参矩阵
        intrinsic_matrix = torch.as_tensor(annotation["intrinsic_matrix"]) # 3x3 
        
        # 读取图像
        image = torchvision.io.read_image(image_filename) # [3,H,W], still is np.uint8
        
        # 绘制掩码
        image = vsrd.visualization.draw_masks(image, masks)
        
        
        # 绘制3D框
        image = vsrd.visualization.draw_boxes_3d(
            image=image,
            boxes_3d=boxes_3d,
            line_indices=LINE_INDICES + [[0, 5], [1, 4]],
            intrinsic_matrix=intrinsic_matrix,
            color=(255, 255, 255),
            thickness=1,
            lineType=cv.LINE_AA,
        )
        
        # write into images
        if not video_writer:
            video_filename = os.path.join(out_dirname, f"{sequence}.mp4")
            os.makedirs(os.path.dirname(video_filename), exist_ok=True)
            video_codec = cv.VideoWriter_fourcc(*"mp4v")
            video_writer = cv.VideoWriter(video_filename, video_codec, frame_rate, image.shape[:2][::-1])

        video_writer.write(image.permute(1, 2, 0).numpy())

        frame_index = int(os.path.splitext(os.path.basename(image_filename))[0])
        image_filename = os.path.join(out_dirname, sequence, f"{frame_index:010}.png")
        os.makedirs(os.path.dirname(image_filename), exist_ok=True)
        torchvision.io.write_png(image, image_filename)

    video_writer.release()



def main(args):
    sequences = list(map(os.path.basename, sorted(glob.glob(os.path.join(args.root_dirname, "data_2d_raw", "*")))))
    # ['2013_05_28_drive_0000_sync', '2013_05_28_drive_0002_sync', '2013_05_28_drive_0003_sync', 
    # '2013_05_28_drive_0004_sync', '2013_05_28_drive_0005_sync', '2013_05_28_drive_0006_sync', 
    # '2013_05_28_drive_0007_sync', '2013_05_28_drive_0009_sync', '2013_05_28_drive_0010_sync']

    with multiprocessing.Pool(args.num_workers) as pool:
        with tqdm.tqdm(total=len(sequences)) as progress_bar:
            # 使用imap_unordered方法异步处理序列
            # 这部分代码使用了Python的multiprocessing模块来实现并行处理。
            # pool.imap_unordered方法将一个可迭代对象（在这里是sequences）分配给多个进程，
            # 以无序的方式并行处理每个元素。imap_unordered返回一个生成器，按处理完成的顺序返回结果，而不是按输入顺序。
            # ------------------------------------------------------------------------------------ #
            #functools.partial用于固定函数的一部分参数，返回一个新的函数对象。
            # 在这里，它固定了visualize_annotations函数的大部分参数，只留下sequence参数未固定。
            # 具体来说，它相当于定义了一个新的函数partial_visualize_annotations(sequence)，这个函数在调用时，
            # 会自动使用root_dirname=args.root_dirname、out_dirname=args.out_dirname、class_names=args.class_names
            # 和frame_rate=args.frame_rate这些固定的参数。
            
            for _ in pool.imap_unordered(functools.partial(
                visualize_annotations,
                root_dirname=args.root_dirname,
                out_dirname=args.out_dirname,
                class_names=args.class_names,
                frame_rate=args.frame_rate,
            ), sequences):
                # 更新进度条
                progress_bar.update(1)


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description="VSRD: Annotation Visualizer for KITTI-360")
    parser.add_argument("--root_dirname", type=str, default="datasets/KITTI-360")
    parser.add_argument("--out_dirname", type=str, default="images/kitti_360/annotations")
    parser.add_argument("--class_names", type=str, nargs="+", default=["car"])
    parser.add_argument("--frame_rate", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=9)
    args = parser.parse_args()

    main(args)
