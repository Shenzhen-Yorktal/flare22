# -*- coding: utf-8 -*-

# 训练第二层网络时，先提取感兴趣区域
import skimage.measure as sm
import SimpleITK as sitk
import nibabel as nb
import numpy as np
import re
import sys
import os
import csv
import configparser
import shutil
import multiprocessing
import time
from multiprocessing.pool import Pool
from nnunet.paths import maybe_mkdir_p2
from batchgenerators.utilities.file_and_folder_operations import *
import threading

# csv保存路径
ROOT_DIR = r''
if not os.path.exists(ROOT_DIR):
    os.makedirs(ROOT_DIR)
# 需要提取ROI的数据文件夹
volume_dir = r''
target_label_dir = r''
bbox_label_dir = r''

# 保存ROI文件夹
roi_volume_dir = r''
roi_label_dir = r''

maybe_mkdir_p2(roi_volume_dir)
maybe_mkdir_p2(roi_label_dir)

# 是否提取最大联通区域
get_lcc = True

bbox_info = []
share_lock = threading.Lock()

def getLargestCC(segmentation):
    if get_lcc:
        labels = sm.label(segmentation)
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        return largestCC
    else:
        return segmentation

def _extract_liver_roi(volume_path, bbox_label_path, target_label_path, padding=[5, 5, 5, 5, 5, 5]):
    '''extract liver roi just for one volume.

    Param:
        volume_path: liver volume path. str
        label_path: liver segmentation path. str
        padding: boarder padding size(x,y.z). int
    return:
        volume_roi: liver region. nibable.image
    '''

    # load data
    image_raw = nb.load(volume_path)
    volume = image_raw.get_fdata()
    bbox_seg_nii = nb.load(bbox_label_path).get_fdata()

    bbox_seg_nii[bbox_seg_nii>0] = 1

    bbox_seg_nii = np.around(bbox_seg_nii).astype('uint8')
    target_seg_nii = nb.load(target_label_path).get_fdata()
    target_seg_nii = np.around(target_seg_nii).astype('uint8')

    shape = volume.shape
    # print('==shape, ', shape)

    # extract object ROI
    box = list(sm.regionprops(bbox_seg_nii)[0].bbox)
    # print('==first box, ', box)
    for i in range(3):
        box[i] = max(0, box[i] - padding[i])
    for i in range(3, 6):
        box[i] = min(box[i] + padding[i], shape[i - 3])
    volume_roi = volume[box[0]:box[3], box[1]:box[4], box[2]:box[5]]
    label_roi = target_seg_nii[box[0]:box[3], box[1]:box[4], box[2]:box[5]]
    # print('==second box, ', box)
    # return nibabel.image
    volume_roi = nb.Nifti1Image(volume_roi, image_raw.affine, header=image_raw.header)
    label_roi = nb.Nifti1Image(label_roi, image_raw.affine, header=image_raw.header)
    return volume_roi, label_roi, box

def extract_roi_parallel(volume_name):
    ''' Extract roi for all volumes in `volume_dir`.'''

    print("Process %s: " % volume_name)

    label_name = volume_name.replace('_0000.nii.gz', '.nii.gz')
    volume_path = os.path.join(volume_dir, volume_name)
    bbox_label_path = os.path.join(bbox_label_dir, label_name)
    target_label_path = os.path.join(target_label_dir, label_name)

    if not os.path.exists(volume_path):
        print('The volume file is not exists: ', volume_path)
        return

    if not os.path.exists(target_label_path):
        print('The target label file is not exists: ', target_label_path)
        return

    if not os.path.exists(bbox_label_path):
        print('The bbox label file is not exists: ', bbox_label_path)
        return

    volume_roi, label_roi, box = _extract_liver_roi(volume_path, bbox_label_path, target_label_path)

    # save results
    volume_roi_path = os.path.join(roi_volume_dir, volume_name)
    nb.save(volume_roi, volume_roi_path)

    label_roi_path = os.path.join(roi_label_dir, label_name)
    nb.save(label_roi, label_roi_path)

    # save a copy as the second channel
    # label_roi_cp_name = volume_name.replace('volume_0000', 'volume_0001')
    # label_roi_cp_path = os.path.join(roi_volume_dir, label_roi_cp_name)
    # nb.save(label_roi, label_roi_cp_path)

    return [label_name, box]

def mycallback(x):
    if x is not None:
        print(x)
        bbox_info.append(x)

if __name__ == "__main__":
    num_processes = 8
    p = Pool(num_processes)

    nii_files = subfiles(volume_dir, suffix=".nii.gz", join=False)
    input_files = [i for i in nii_files]

    results = [p.apply_async(extract_roi_parallel, (i,), callback=mycallback) for i in input_files]

    p.close()
    p.join()

    # save coordinate
    csv_path = os.path.join(ROOT_DIR, 'coord.csv')
    if os.path.exists(csv_path):
        csv_file = open(csv_path, 'a', newline='')
    else:
        csv_file = open(csv_path, 'w')

    eWriter = csv.writer(csv_file, delimiter=',', lineterminator='\n')  #
    for row in bbox_info:
        eWriter.writerow(row)

    csv_file.close()





