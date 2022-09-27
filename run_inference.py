"""
This python script is a dummy example of the inference script that populates output/ folder. For this example, the
loading of the model from the directory /model is not taking place and the output/ folder is populated with arrays
filled with zeros of the same size as the images in the input/ folder.
"""

import os
import numpy as np
import torch

import skimage.measure as sm
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, isfile, subdirs
# from nnunet.inference.predict import predict_cases
from predict_1by1 import predict_cases
from time import time

def clear_cpu():
    try:
        import gc
        gc.collect()
    except:
        return

def maybe_create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    # outpth = Path(dir)
    # outpth.mkdir(parents=True, exist_ok=True)

INPUT_NIFTI = '/workspace/inputs'
OUTPUT_NIFTI = '/workspace/outputs'
parameter_folder = '/parameters'

if os.path.exists('/home/xudong/flare22/docker/parameters'):
    # INPUT_NIFTI = '/home/xudong/flare22/val_test'
    # OUTPUT_NIFTI = '/home/xudong/flare22/val_test_out'
    INPUT_NIFTI = '/home/xudong/flare22/ValidationSet'
    OUTPUT_NIFTI = '/home/xudong/flare22/ValidationSetInfer'
    parameter_folder = '/home/xudong/flare22/docker/parameters'

maybe_create_dir(INPUT_NIFTI)
maybe_create_dir(OUTPUT_NIFTI)

PADDING = [20, 20, 20, 80, 30, 80]
#infer roi
use_gaussian = True  # True
do_tta = False

# def get_num_processes():
#     return 1


def getLargestCC(segmentation):
    labels = sm.label(segmentation, connectivity=2)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC.astype('uint8')

# def __extract_largest_roi(label, img_dir, label_dir, img_dir_save, label_dir_save, padding=[20, 20, 20, 80, 30, 20]):
#     img_file = join(img_dir, label)
#     img_itk = sitk.ReadImage(img_file, sitk.sitkFloat32)
#     image_array = sitk.GetArrayFromImage(img_itk)
#     shape = image_array.shape
#
#     label_file = join(label_dir, label)
#     label_itk = sitk.ReadImage(label_file, sitk.sitkUInt8)
#     label_data = sitk.GetArrayFromImage(label_itk)
#     label_data = getLargestCC(label_data)
#
#     # extract object ROI
#     box = list(sm.regionprops(label_data)[0].bbox)
#     # box_dict[label] = [shape, box]
#     for i in range(3):
#         box[i] = max(0, box[i] - padding[i])
#     for i in range(3, 6):
#         box[i] = min(box[i] + padding[i], shape[i - 3])
#     image_roi = image_array[box[0]:box[3], box[1]:box[4], box[2]:box[5]]
#     label_roi = label_data[box[0]:box[3], box[1]:box[4], box[2]:box[5]]
#
#     img_roi_itk = sitk.GetImageFromArray(image_roi)
#     label_roi_itk = sitk.GetImageFromArray(label_roi)
#     img_roi_itk.SetSpacing(img_itk.GetSpacing())
#     img_roi_itk.SetOrigin(img_itk.GetOrigin())
#     img_roi_itk.SetDirection(img_itk.GetDirection())
#
#     label_roi_itk.SetOrigin(img_itk.GetOrigin())
#     label_roi_itk.SetOrigin(img_itk.GetOrigin())
#     label_roi_itk.SetDirection(img_itk.GetDirection())
#
#     sitk.WriteImage(img_roi_itk, join(img_dir_save, label))
#     sitk.WriteImage(label_roi_itk, join(label_dir_save, label))
#
#     return {label: [shape, box]}

def __extract_largest_roi(label, img_dir, label_dir, img_dir_save, label_dir_save, padding=PADDING):
    img_file = join(img_dir, label)
    img_itk = sitk.ReadImage(img_file, sitk.sitkFloat32)
    image_array = sitk.GetArrayFromImage(img_itk)
    shape = image_array.shape

    label_file = join(label_dir, label)
    label_itk = sitk.ReadImage(label_file, sitk.sitkUInt8)
    label_data = sitk.GetArrayFromImage(label_itk)
    spacing = label_itk.GetSpacing()
    print('spcing: ', label_file, spacing)
    if spacing[2] > 1.0:
        print('befor: ', padding)
        padding[0] = int(40 / spacing[2] + 1)
        padding[3] = int(120 / spacing[2] + 1)
        print('after: ', padding)

    labels, num = sm.label(label_data, return_num=True, connectivity=2)
    # print('area: ', label, num)
    if num > 0:
        props = sm.regionprops(labels)
        area = {ele.area: ele for ele in props}

        area = sorted(area.items(), key=lambda kv: (kv[0], kv[1]), reverse=True)
        # max_area = area[0][0]
        # print('max area: ', max_area)
        label_data = labels == area[0][1].label
        # label_data = label_data1
        # for i in range(0, num):
        #     src_area = props[i].area
        #     if src_area / max_area > radio:
        #         # print('add: ', i, src_area)
        #         label_data2 = labels == props[i].label
        #         label_data = label_data + label_data2
        label_data = label_data.astype('uint8')

        # extract object ROI
        box = list(area[0][1].bbox)
        # box_dict[label] = [shape, box]
        for i in range(3):
            box[i] = max(0, box[i] - padding[i])
        for i in range(3, 6):
            box[i] = min(box[i] + padding[i], shape[i - 3])
        image_roi = image_array[box[0]:box[3], box[1]:box[4], box[2]:box[5]]
        label_roi = label_data[box[0]:box[3], box[1]:box[4], box[2]:box[5]]

        img_roi_itk = sitk.GetImageFromArray(image_roi)
        label_roi_itk = sitk.GetImageFromArray(label_roi)
        img_roi_itk.SetSpacing(img_itk.GetSpacing())
        img_roi_itk.SetOrigin(img_itk.GetOrigin())
        img_roi_itk.SetDirection(img_itk.GetDirection())

        label_roi_itk.SetOrigin(img_itk.GetOrigin())
        label_roi_itk.SetOrigin(img_itk.GetOrigin())
        label_roi_itk.SetDirection(img_itk.GetDirection())

        sitk.WriteImage(img_roi_itk, join(img_dir_save, label))
        sitk.WriteImage(label_roi_itk, join(label_dir_save, label))
    else:
        print('!!!error, no regionprops: ', label_file)
        sitk.WriteImage(img_itk, join(img_dir_save, label))
        sitk.WriteImage(label_itk, join(label_dir_save, label))
        box = [0, 0, 0, shape[0], shape[1], shape[2]]
        for i in range(3):
            box[i] = max(0, box[i] - padding[i])
        for i in range(3, 6):
            box[i] = min(box[i] + padding[i], shape[i - 3])

    return {label: [shape, box]}


def extract_largest_roi(box_dict, img_dir, label_dir, img_dir_save, label_dir_save, padding=PADDING, radio=0.05):
    label_files = subfiles(label_dir, suffix='.nii.gz', join=False)
    # from multiprocessing.pool import Pool
    # p = Pool(get_num_processes())
    # results = [
    #     p.apply_async(__extract_largest_roi, (label, img_dir, label_dir, img_dir_save, label_dir_save, padding, radio))
    #     for label in label_files]
    # p.close()
    # p.join()
    # for r in results:
    #     box_dict.update(r.get())
    for label in label_files:
        r = __extract_largest_roi(label, img_dir, label_dir, img_dir_save, label_dir_save)
        print(r)
        box_dict.update(r)
        clear_cpu()
    # print(len(box_dict))


def get_LCC(input_dir, output_dir):
    from nnunet.postprocessing.connected_components import apply_postprocessing_to_folder
    for_which_classes = [1, 2, 3, 5, 7, 8, 13]
    apply_postprocessing_to_folder(input_dir, output_dir, for_which_classes, num_processes=1)


def putback_roi(shape, coord, mask_file, save_file):
    if shape is None:
        return None
    data_array = np.zeros(shape, dtype='uint8')
    spacing = [1, 1, 1]
    origin = [0, 0, 0]
    direction = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    if isfile(mask_file):
        img_itk = sitk.ReadImage(mask_file, sitk.sitkUInt8)
        spacing = img_itk.GetSpacing()
        origin = img_itk.GetOrigin()
        direction = img_itk.GetDirection()
        img_data = sitk.GetArrayFromImage(img_itk)
        data_array[coord[0]:coord[3], coord[1]:coord[4], coord[2]:coord[5]] = img_data

    data_itk = sitk.GetImageFromArray(data_array)
    data_itk.SetSpacing(spacing)
    data_itk.SetOrigin(origin)
    data_itk.SetDirection(direction)
    sitk.WriteImage(data_itk, save_file.replace('_0000', ''))
    # return data_array, spacing, origin, direction


def get_final_result(output_dir, roi_dir, box_dicts):
    for key, value in box_dicts.items():
        # print(key, value)
        mask_roi = join(roi_dir, key)
        save_file = join(output_dir, key)
        putback_roi(value[0], value[1], mask_roi, save_file)


def main():
    start = time()
    # this will be changed to /input for the docker
    input_folder = INPUT_NIFTI

    # this will be changed to /output for the docker
    output_folder = OUTPUT_NIFTI
    output_folder_lowres = join(OUTPUT_NIFTI, 'lowres_output')
    maybe_create_dir(output_folder_lowres)

    input_files = subfiles(input_folder, suffix='.nii.gz', join=False)

    output_files_lowres = [join(output_folder_lowres, i) for i in input_files]
    input_files_lowres = [join(input_folder, i) for i in input_files]

    # in the parameters folder are five models (fold_X) traines as a cross-validation. We use them as an ensemble for
    # prediction
    folds = 0

    # setting this to True will make nnU-Net use test time augmentation in the form of mirroring along all axes. This
    # will increase inference time a lot at small gain, so you can turn that off

    # does inference with mixed precision. Same output, twice the speed on Turing and newer. It's free lunch!
    mixed_precision = True
    parameter_folder_lowres = os.path.join(parameter_folder, 'Task026_FLARE22')
    predict_cases(parameter_folder_lowres, [[i] for i in input_files_lowres], output_files_lowres, folds, save_npz=False,
                  num_threads_preprocessing=1, num_threads_nifti_save=1, segs_from_prev_stage=None, do_tta=False,
                  mixed_precision=mixed_precision, overwrite_existing=True, all_in_gpu=False, step_size=0.75,
                  use_gaussian=False)

    clear_cpu()
    t1 = time()
    print('time for lowres: ', t1-start)
    ###########################
    #extract ROI
    img_lowres_roi = join(output_folder, 'img_lowres_roi')
    label_lowres_roi = output_folder_lowres + '_roi'
    maybe_create_dir(img_lowres_roi)
    maybe_create_dir(label_lowres_roi)

    box_dict_lowres = {}
    print('extract roi')
    extract_largest_roi(box_dict_lowres, input_folder, output_folder_lowres, img_lowres_roi, label_lowres_roi)
    clear_cpu()
    t2 = time()
    print('time for extract roi: ', t2 - t1)

    ########
    #infer ROI

    output_folder_roi = join(output_folder, 'output_roi')
    maybe_create_dir(output_folder_roi)
    output_files_roi = [join(output_folder_roi, i) for i in input_files]
    input_files_roi = [join(img_lowres_roi, i) for i in input_files]

    parameter_folder_roi = os.path.join(parameter_folder, 'Task027_FLARE22')
    folds = 'all'
    print('infer roi')
    predict_cases(parameter_folder_roi, [[i] for i in input_files_roi], output_files_roi, folds, save_npz=False,
                  num_threads_preprocessing=1, num_threads_nifti_save=1, segs_from_prev_stage=None, do_tta=do_tta,
                  mixed_precision=mixed_precision, overwrite_existing=True, all_in_gpu=False, step_size=0.5,
                  use_gaussian=use_gaussian)

    t3 = time()
    print('time for fullres: ', t3 - t2)
    output_folder_roi_ppd = output_folder_roi + '_ppd'
    get_LCC(output_folder_roi, output_folder_roi_ppd)


    get_final_result(output_folder, output_folder_roi_ppd, box_dict_lowres)
    dirs = [d for d in os.listdir(output_folder) if not d.endswith('.nii.gz')]
    import shutil
    # shutil.rmtree(output_folder_lowres)
    # shutil.rmtree(img_lowres_roi)
    # shutil.rmtree(label_lowres_roi)
    # shutil.rmtree(output_folder_roi)
    # shutil.rmtree(output_folder_roi_ppd)
    if not os.path.exists('/home/xudong/flare22/docker/parameters'):
        for dir in dirs:
            dir_path = os.path.join(output_folder, dir)
            print('remove temp dir:', dir_path)
            if os.path.isdir(dir_path):
                shutil.rmtree(dir_path)
            elif os.path.isfile(dir_path):
                os.remove(dir_path)

    print('done! all time: ', time() - start)


if __name__ == '__main__':
    main()
