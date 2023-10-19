
# ======================================================================================================================
# Verifies that the extracted image rotation matrix, bounding box coordinates,
# and final transformation matrix are correct.
# ======================================================================================================================

from PIL import Image
import numpy as np
import json
import cv2
import skimage
import imageio
import glob
from pathlib import Path
import os
import json
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

dir_img = "Z:/Public/Jonas/Data/ESWW007/SingleLeaf"
dir_mask = "Z:/Public/Jonas/Data/ESWW007/SingleLeaf/Inference/Mask"
dir_meta = "Z:/Public/Jonas/Data/ESWW007/SingleLeaf/Output"
masks = glob.glob(f'{dir_mask}/*ESWW0070020_1.png')


for m in masks:

    print(m)

    # get names
    base_name = os.path.basename(m)
    jpg_name = base_name.replace(".png", ".JPG")
    stem_name = base_name.replace(".png", "")
    leaf_name = "_".join(stem_name.split("_")[2:4])
    date = stem_name.split("_")[0]

    # get mask
    mask = Image.open(m)
    mask = np.asarray(mask)

    # get image
    try:
        image = f'{dir_img}/{date}/JPEG_cam/{jpg_name}'
        img = Image.open(image)
    except FileNotFoundError:
        try:
            image = f'{dir_img}/{date}_1/JPEG_cam/{jpg_name}'
            img = Image.open(image)
        except FileNotFoundError:
            image = f'{dir_img}/{date}_2/JPEG_cam/{jpg_name}'
            img = Image.open(image)
    img = np.asarray(img)
    plt.imshow(img)

    # get the target (warped roi)
    target = Image.open(f"{dir_meta}/{leaf_name}/result/{stem_name}.JPG")
    target = np.asarray(target)

    # get bounding box localization info
    output_p = f"{dir_meta}/{leaf_name}/roi/{stem_name}.json"
    f = open(output_p)
    data = json.load(f)
    rot = np.asarray(data['rotation_matrix'])
    bbox = np.asarray(data['bounding_box'])
    box_ = np.intp(bbox)
    tform = None
    if "transformation_matrix" in [k for k in data.keys()]:
        tform = np.asarray(data['transformation_matrix'])
    f.close()

    # get roi
    rows, cols = img.shape[0], img.shape[1]
    img_rot = cv2.warpAffine(img, rot, (cols, rows))
    ll = img_rot[box_[0][1]:box_[2][1], box_[0][0]:box_[1][0]]

    # remove the padding
    height = ll.shape[0] + 100
    margin = 32 - (height % 32)
    leaf = mask[margin:mask.shape[0], :]

    # shrink to actual roi
    leaf = leaf[50:leaf.shape[0]-50, box_[0][0]:box_[1][0]]

    out_dir = Path(f'{dir_meta}/{leaf_name}/mask_aligned/')
    if not out_dir.exists():
        out_dir.mkdir(exist_ok=True, parents=True)
    mask_name = f'{out_dir}/{base_name}'

    if tform is not None:
        warped_m = skimage.transform.warp(leaf, tform, output_shape=target.shape)
        warped_m = skimage.util.img_as_ubyte(warped_m)

        # perform the transformation
        warped = skimage.transform.warp(ll, tform, output_shape=target.shape)
        warped = skimage.util.img_as_ubyte(warped)
        imageio.imwrite(mask_name, warped_m)
    else:
        imageio.imwrite(mask_name, leaf)

    tform = None

    # # test
    # fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    # axs[0].imshow(ll)
    # axs[0].set_title('leaf')
    # axs[1].imshow(warped)
    # axs[1].set_title('warped')
    # axs[2].imshow(warped_m)
    # axs[2].set_title('target')
    # plt.show(block=True)
    #
    # warped_m.shape
    # warped.shape
    # target.shape

    # diff = np.abs(warped[:, :, 0]+100 - target[:, :, 0])
    # plt.imshow(diff)
    # # OK!!


# ======================================================================================================================

# check the output

from pathlib import Path
import itertools

# def list_by_sample(path_rois, path_images):
#
#     roi_series = []
#     image_series = []
#
#     labels = glob.glob(f'{path_rois}/*.json')
#     images = glob.glob(f'{path_images}/*.JPG')
#     label_image_id = ["_".join(os.path.basename(l).split("_")[2:4]).replace(".txt", "") for l in labels]
#     image_image_id = ["_".join(os.path.basename(l).split("_")[2:4]).replace(".JPG", "") for l in images]
#     uniques = np.unique(label_image_id)
#
#     len(images)
#     len(labels)
#
#     # if len(images) != len(labels):
#         # raise Exception("list of images and list of coordinate files are not of equal length.")
#         # warnings.warn("list of images and list of coordinate files are not of equal length."
#         #               "Ignoring extra coordinate files.")
#
#     print("found " + str(len(uniques)) + " unique sample names")
#
#     for unique_sample in uniques:
#         image_idx = [index for index, image_id in enumerate(image_image_id) if unique_sample == image_id]
#         label_idx = [index for index, label_id in enumerate(label_image_id) if unique_sample == label_id]
#         sample_image_names = [images[i] for i in image_idx]
#         sample_labels = [labels[i] for i in label_idx]
#         roi_series.append(sample_labels)
#         image_series.append(sample_image_names)
#
#     return roi_series, image_series
#
# r_list, i_list = list_by_sample(path_rois='Z:/Public/Jonas/Data/ESWW007/SingleLeaf/Output/*/roi',
#                                 path_images='Z:/Public/Jonas/Data/ESWW007/SingleLeaf/*/JPEG_cam')


data_root = 'Z:/Public/Jonas/Data/ESWW007/SingleLeaf/'
roi_root = 'Z:/Public/Jonas/Data/ESWW007/SingleLeaf/Output/'

# path = Path(data_root)
# path_roi = Path(roi_root)


def list_by_date(data_root, roi_root):

    dates = glob.glob(f'{data_root}/2023*')
    dates = [os.path.basename(x) for x in dates]

    rois = [x for x in glob.glob(f"{roi_root}/*/roi/*.json")]

    JPG = []
    ROI = []
    for d in dates:

        # get all image paths
        jpg_paths = glob.glob(f"{data_root}/{d}/JPEG_cam/*.JPG")
        # get file names
        jpg_files = [os.path.basename(x).replace(".JPG", "") for x in jpg_paths]

        # get corresponding roi paths
        roi_paths = []
        for b in jpg_files:
            a = [x for x in rois if b in x]
            roi_paths.append(a)
        roi_paths = [item for sublist in roi_paths for item in sublist]
        roi_files = [os.path.basename(x).replace(".json", "") for x in roi_paths]

        # ignore image paths, if no corresponding roi path is found
        roi_idx = [index for index, roi_id in enumerate(roi_files) if roi_id in jpg_files]
        roi_files = [roi_paths[i] for i in roi_idx]

        JPG.append(jpg_paths)
        ROI.append(roi_files)

    return JPG, ROI









    JPGs.append(JPG)
    JSONs.append(roi_files)

date_images, date_rois = list_by_date(data_root, roi_root)

for d_img, d_roi in zip(date_images, date_rois):

    if len(d_img) != len(d_roi):
        print("series not of equal length!")
        print(d_img)
        print(d_roi)
        break

    leaf_crops = []
    image_ids = []
    for j in range(len(d_img)):

        # get image id
        image_id = os.path.basename(d_img[j]).replace(".JPG", "")

        # get img
        img = Image.open(d_img[j])
        img = np.asarray(img)

        # get roi coordinates and rotation
        f = open(d_roi[j])
        data = json.load(f)
        rot = np.asarray(data['rotation_matrix'])
        bbox = np.asarray(data['bounding_box'])
        f.close()

        # rotate the image
        rows, cols = img.shape[0], img.shape[1]
        img_rot = cv2.warpAffine(img, rot, (cols, rows))

        # get the leaf
        box_ = np.intp(bbox)
        leaf = img_rot[box_[0][1] - 50:box_[2][1] + 50, 0:8192]

        leaf_crops.append(leaf)
        image_ids.append(image_id)




