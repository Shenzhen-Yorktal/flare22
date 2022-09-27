import nibabel as nb
import numpy as np
import os
import csv

# 提取的ROI坐标
csv_path = r'E:\testdata\flare22\coord.csv'

# ROI预测结果路径
roi_inference_dir = r''

# 原始volume路径
volume_dir = r''

# roi预测结果，补全之后的保存路径
roi_put_back_dir = r''

def csv_process(csv_path):
    '''convert csv data to dict.

    Param: csv_path:
        csv file path.
    return: location: dict
        the rectangle box data.
    '''
    efile = open(csv_path, 'r')
    eReader = csv.reader(efile)
    location = dict()

    for row in eReader:
        # print(row)
        box = row[1][1:len(row[1]) - 1].replace(' ', '').split(',')
        box = [int(temp) for temp in box]
        location[row[0]] = box
    return location

def post_process(volume_path, mask_path, coord):
    '''
    Param:
        volume_path: original liver volume path. str
        mask_path: the path ROI segmentation result. str
        coord: the ROI location. list
    return:
        label: nb.image
    '''
    volume_nii = nb.load(volume_path)
    volume = volume_nii.get_fdata()
    mask_nii = nb.load(mask_path)
    mask = mask_nii.get_fdata()
    mask = np.around(mask).astype('uint8')
    mask = np.squeeze(mask)

    label = np.zeros_like(volume).astype('uint8')
    label[coord[0]:coord[3], coord[1]:coord[4], coord[2]:coord[5]] = mask

    label_nii = nb.Nifti1Image(label, mask_nii.affine, header=mask_nii.header)
    return label_nii


def do_or(label, roi):
    if isinstance(label, str):
        if os.path.isfile(label) and (label.endswith('.nii.gz') or label.endswith('.nii')):
            label = nb.load(label).get_data()
    if isinstance(roi, str):
        if os.path.isfile(roi) and (roi.endswith('.nii.gz') or roi.endswith('.nii')):
            roi = nb.load(roi)
    if isinstance(label, np.ndarray) and isinstance(roi, nb.nifti1.Nifti1Image):
        result = label | roi.get_data()
        result = nb.Nifti1Image(result, roi.affine, header=roi.header)
        return result
    return None


def put_roi_back():
    volume_list = os.listdir(volume_dir)
    # csv_path = os.path.join(r'H:\Urinary-data\result\left-kidney\test\nnunet-infer', 'left-kidney-coord.csv')
    location = csv_process(csv_path)  # dict
    print('get csv done!')

    for volume_name in volume_list:
        name = volume_name.split('.')[0].replace('_0000', '')
        # name = volume_name.split('.')[0]
        volume_path = os.path.join(volume_dir, volume_name)
        seg_path = os.path.join(roi_inference_dir, name + '.nii.gz')
        if not os.path.exists(seg_path):
            continue
        roi = post_process(volume_path, seg_path, location[name])
        save_path = os.path.join(roi_put_back_dir, name + '.nii.gz')
        nb.save(roi, save_path)
        print('Save {}.'.format(save_path))

if __name__ == "__main__":
    if not os.path.isdir(roi_put_back_dir):
        os.makedirs(roi_put_back_dir)

    put_roi_back()
