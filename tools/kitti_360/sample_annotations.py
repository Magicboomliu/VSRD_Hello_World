import os
import json
import glob
import operator
import argparse
import functools
import itertools
import collections
import multiprocessing

import tqdm
import numpy as np
import pycocotools.mask


def sample_annotations(
    sequence,
    root_dirname,
    class_names,
    num_instance_ratio,
    num_source_frames,
    min_mask_area,
    min_box_size,
):
    
    
    # get all the images
    image_filenames = sorted(glob.glob(os.path.join(root_dirname, "data_2d_raw", sequence, "image_00", "data_rect", "*.png")))
    # get all teh frames idx
    # os.path.splitext() 函数用来将文件路径分割成一个元组 (root, ext), which is expressed in int
    frame_indices = [
        int(os.path.splitext(os.path.basename(image_filename))[0])
        for image_filename in image_filenames
    ] # get all the image ids

    min_frame_index = min(frame_indices) # the most begininng
    max_frame_index = max(frame_indices) # the most ending

    #在 Python 中是用来初始化一个带有默认值类型为列表的 defaultdict。
    # 这种字典属于 collections 模块，并且在你希望字典对于尚未存在的键自动赋予一个默认值时非常有用。
    # 当你使用一个不存在的键来访问或修改字典时，defaultdict 会自动创建一个新的条目，并使用在创建时提供的函数来初始化它，本例中使用的是 list。
    grouped_image_filenames = collections.defaultdict(list)

    print(len(image_filenames))
    for target_image_filename in tqdm.tqdm(image_filenames):

        def sample_source_frames(target_annotation_filename):
            if not os.path.exists(target_annotation_filename): return [], []
            with open(target_annotation_filename) as file:
                target_annotation = json.load(file)

            def check_mask(mask):
                # mask is [H,W]
                mask_area = np.sum(mask) # 
                indices = np.where(mask) 
                min_indices = list(map(min, indices))
                max_indices = list(map(max, indices))
                box_size = np.min(np.subtract(max_indices, min_indices))
                return (mask_area >= min_mask_area) & (box_size >= min_box_size)

            # 在原始代码中使用的 sum 函数不是用于数值加和，
            # 而是用于合并多个列表。在 Python 中，sum 函数可以接受两个参数：一个可迭代对象（通常是列表的列表）和一个起始值，这个起始值在这里用于确定合并的数据类型和初始状态。
            # get all possible instances ids in the target frame: return a list

            target_instance_ids = sum([
                [
                    instance_id
                    for instance_id, mask in masks.items()
                    if check_mask(pycocotools.mask.decode(mask))
                ]
                for class_name, masks
                in target_annotation["masks"].items()
                if class_name in class_names
            ], [])
            
            if not target_instance_ids: return [], []

            # get the source realtive indices
            source_relative_indices = []
            # get the target frame idx
            target_frame_index = int(os.path.splitext(os.path.basename(target_annotation_filename))[0])

            for source_relative_index in itertools.count(1):

                source_frame_index = target_frame_index + source_relative_index # source faame

                # look until to the end
                if max_frame_index < source_frame_index: break
                # get the json files
                source_annotation_filename = os.path.join(os.path.dirname(target_annotation_filename), f"{source_frame_index:010}.json")
                # make sure the file names is exited
                if not os.path.exists(source_annotation_filename): continue

                
                with open(source_annotation_filename) as file:
                    source_annotation = json.load(file)

                #  get the instances id contains
                source_instance_ids = sum([
                    [
                        instance_id
                        for instance_id, mask in masks.items()
                        if check_mask(pycocotools.mask.decode(mask))
                    ]
                    for class_name, masks
                    in source_annotation["masks"].items()
                    if class_name in class_names
                ], [])

                if len(set(target_instance_ids) & set(source_instance_ids)) / len(target_instance_ids) < num_instance_ratio: break

                source_relative_indices.append(source_relative_index)

            for source_relative_index in itertools.count(1):
                
                # look in the previous
                source_frame_index = target_frame_index - source_relative_index

                # look though all the 
                if source_frame_index < min_frame_index: break

                source_annotation_filename = os.path.join(os.path.dirname(target_annotation_filename), f"{source_frame_index:010}.json")

                if not os.path.exists(source_annotation_filename): continue

                with open(source_annotation_filename) as file:
                    source_annotation = json.load(file)

                source_instance_ids = sum([
                    [
                        instance_id
                        for instance_id, mask in masks.items()
                        if check_mask(pycocotools.mask.decode(mask))
                    ]
                    for class_name, masks
                    in source_annotation["masks"].items()
                    if class_name in class_names
                ], [])

                if len(set(target_instance_ids) & set(source_instance_ids)) / len(target_instance_ids) < num_instance_ratio: break

                source_relative_indices.append(-source_relative_index)
            
            return sorted(target_instance_ids), sorted(source_relative_indices)

        target_annotation_filename = target_image_filename.replace("data_2d_raw", "annotations").replace(".png", ".json")
        # return two types: the target instance ids and tyhe source relative indices
        target_instance_ids, source_relative_indices = sample_source_frames(target_annotation_filename)
       

        if len(source_relative_indices) >= num_source_frames:
            grouped_image_filenames[tuple(target_instance_ids)].append((target_image_filename, source_relative_indices))

    # saved match image filenames
    grouped_image_filename = os.path.join(
        root_dirname,
        f"filenames",
        f"R{num_instance_ratio * 100.0:.0f}-"
        f"N{num_source_frames}-"
        f"M{min_mask_area}-"
        f"B{min_box_size}",
        sequence,
        "grouped_image_filenames.txt"
    )
    

    sampled_image_filename = os.path.join(
        root_dirname,
        f"filenames",
        f"R{num_instance_ratio * 100.0:.0f}-"
        f"N{num_source_frames}-"
        f"M{min_mask_area}-"
        f"B{min_box_size}",
        sequence,
        "sampled_image_filenames.txt"
    )

    os.makedirs(os.path.dirname(grouped_image_filename), exist_ok=True)
    os.makedirs(os.path.dirname(sampled_image_filename), exist_ok=True)

    with open(grouped_image_filename, "w") as grouped_image_file:
        with open(sampled_image_filename, "w") as sampled_image_file:

            for target_instance_ids, grouped_image_filenames in grouped_image_filenames.items():

                grouped_image_filenames = sorted(grouped_image_filenames, 
                                                 key=lambda item: int(os.path.splitext(os.path.basename(item[0]))[0]))
                
                # Make sure all instance ids have only one tracking, one traget image
                target_image_filename, source_relative_indices = grouped_image_filenames[len(grouped_image_filenames) // 2]
                grouped_image_file.write(f"{','.join(target_instance_ids)} {','.join(map(operator.itemgetter(0), grouped_image_filenames))}\n")
                sampled_image_file.write(f"{','.join(target_instance_ids)} {target_image_filename} {','.join(map(str, source_relative_indices))}\n")




def main(args):

    sequences = list(map(os.path.basename, sorted(glob.glob(os.path.join(args.root_dirname, "data_2d_raw", "*")))))

    
    with multiprocessing.Pool(args.num_workers) as pool:
        with tqdm.tqdm(total=len(sequences)) as progress_bar:

            for _ in pool.imap_unordered(functools.partial(
                sample_annotations,
                root_dirname=args.root_dirname,
                class_names=args.class_names,
                num_instance_ratio=args.num_instance_ratio,
                num_source_frames=args.num_source_frames,
                min_mask_area=args.min_mask_area,
                min_box_size=args.min_box_size,
            ), sequences):

                progress_bar.update(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="VSRD: Annotation Sampler for KITTI-360")
    parser.add_argument("--root_dirname", type=str, default="datasets/KITTI-360")
    parser.add_argument("--class_names", type=str, nargs="+", default=["car"])
    parser.add_argument("--num_instance_ratio", type=float, default=0.5) # should cover more than 500 images
    parser.add_argument("--num_source_frames", type=int, default=16) # how many source fames
    parser.add_argument("--min_mask_area", type=int, default=128)   # instance mask h*w should bigger than 128
    parser.add_argument("--min_box_size", type=int, default=16)     # height or width should bigger than 16
    parser.add_argument("--num_workers", type=int, default=9)
    args = parser.parse_args()

    main(args)
