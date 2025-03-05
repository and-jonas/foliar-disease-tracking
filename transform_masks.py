
# ======================================================================================================================
# Verifies that the extracted image rotation matrix, bounding box coordinates,
# and final transformation matrix are correct.
# Crops the roi, rotates and translates roi, transforms detected points,
# and outputs the rectified masks for further processing.
# ======================================================================================================================

from PIL import Image
import numpy as np
import cv2
import pickle
import imageio
import skimage
import glob
from pathlib import Path
import os
import json
import utils
from natsort import natsorted
import multiprocessing
from tqdm import tqdm

# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('Qt5Agg')

# mask_dir = "Z:/Public/Jonas/Data/ESWW009/SingleLeaf/predictions"
# output_dir = "Z:/Public/Jonas/Data/ESWW009/SingleLeaf/Output"
mask_dir = "/home/anjonas/public/Public/Jonas/Data/ESWW009/SingleLeaf/predictions"
output_dir = "/home/anjonas/public/Public/Jonas/Data/ESWW009/SingleLeaf/Output"

# list all samples for which the transformation was successful
existing_output = glob.glob(f'{output_dir}/*/result/piecewise/*.JPG')
bnames = [os.path.basename(x).replace(".JPG", "") for x in existing_output]

# list all paths to masks
# masks = glob.glob(f'{output_dir}/*/mask2/*.png')
# masks = glob.glob(f'{output_dir}/*/mask/*.png')
masks = glob.glob(f'{mask_dir}/*.png')

label_image_id = ["_".join(os.path.basename(l).split("_")[2:4]).replace(".txt", "") for l in masks]
uniques = natsorted(np.unique(label_image_id))

# compile the lists
label_series = []
for unique_sample in uniques:
    label_idx = [index for index, label_id in enumerate(label_image_id) if unique_sample == label_id]
    sample_labels = [masks[i] for i in label_idx]
    # sort to ensure sequential processing of subsequent images
    sample_labels = sorted(sample_labels, key=lambda i: os.path.splitext(os.path.basename(i))[0])
    label_series.append(sample_labels)
masks = [l for ls in label_series for l in ls]
masks = [m for m in masks if os.path.basename(m).replace(".png", "") in bnames]

# remove processed from list
processed = glob.glob(f'{output_dir}/*/mask_aligned/piecewise/*.png')
pnames = [os.path.basename(x).replace(".png", "") for x in processed]
masks = [m for m in masks if os.path.basename(m).replace(".png", "") not in pnames]

# ======================================================================================================================


# Process masks
def transform_mask(output_dir, path_to_mask, n_classes, kpt_cls):

    print(path_to_mask)

    # get names
    base_name = os.path.basename(path_to_mask)
    jpg_name = base_name.replace(".png", ".JPG")
    stem_name = base_name.replace(".png", "")
    leaf_name = "_".join(stem_name.split("_")[2:4])

    # get mask
    mask = Image.open(path_to_mask)
    mask = np.asarray(mask)

    # # get image
    # image = f'{base_dir}/{leaf_name}/crop/{jpg_name}'
    # img = Image.open(image)
    # img = np.asarray(img)

    # ==================================================================================================================

    # get the target (warped roi)
    target = Image.open(f"{output_dir}/{leaf_name}/result/piecewise/{stem_name}.JPG")
    target = np.asarray(target)

    # get bounding box localization info
    output_p = f"{output_dir}/{leaf_name}/roi/{stem_name}.json"
    f = open(output_p)
    data = json.load(f)
    rot = np.asarray(data['rotation_matrix'])
    bbox = np.asarray(data['bounding_box'])
    box = np.intp(bbox)

    tform_piecewise = None
    try:
        with open(f'{output_dir}/{leaf_name}/roi/{stem_name}_tform_piecewise.pkl', 'rb') as file:
            tform_piecewise = pickle.load(file)
    except FileNotFoundError:
        pass

    # get image roi
    # full_img = np.zeros((5464, 8192, 3)).astype("uint8")
    mw, mh = map(int, np.mean(box, axis=0))
    # full_img[mh-1024:mh+1024, :] = img
    # rows, cols = full_img.shape[0], full_img.shape[1]
    rows, cols = 5464, 8192

    # full mask
    full_mask = np.zeros((5464, 8192)).astype("uint8")
    full_mask[mh - 1024:mh + 1024, :] = mask

    # ==================================================================================================================
    # Transform the mask
    # ==================================================================================================================

    # get rid of the points
    segmentation_mask = utils.remove_points_from_mask(mask=full_mask, classes=kpt_cls)

    # rotate mask
    segmentation_mask_rot = cv2.warpAffine(segmentation_mask, rot, (cols, rows))

    # crop roi
    roi = segmentation_mask_rot[box[0][1]:box[2][1], box[0][0]:box[1][0]]

    # warp roi (except for the first image in the series)
    tform = tform_piecewise
    if tform is not None:
        lm = np.stack([roi, roi, roi], axis=2)
        warped = skimage.transform.warp(lm, tform, output_shape=target.shape)
        warped = skimage.util.img_as_ubyte(warped[:, :, 0])
    else:
        warped = roi

    # ==================================================================================================================
    # Transform points, add to mask
    # ==================================================================================================================

    # warp points
    if tform is not None:
        complete = utils.rotate_translate_warp_points(
            mask=full_mask,
            classes=kpt_cls,
            rot=rot,
            box=box,
            tf=tform,
            target_shape=target.shape,
            warped=warped,
        )
    else:
        complete = warped

    # ==================================================================================================================
    # Output
    # ==================================================================================================================

    # transform to ease inspection
    complete = (complete.astype("uint32")) * 255 / n_classes
    complete = complete.astype("uint8")

    # save
    out_dir = Path(f'{output_dir}/{leaf_name}/mask_aligned/')
    out_dir_pw = out_dir / "piecewise"
    out_dir_pw.mkdir(exist_ok=True, parents=True)
    mask_name = f'{out_dir_pw}/{base_name}'
    imageio.imwrite(mask_name, complete)


if __name__ == '__main__':

    # get number of samples to process
    n = len(masks)

    # list tasks
    output_dir = [output_dir] * n
    n_classes = [6] * n
    kpt_cls = [(5, 6)] * n
    tasks = [*zip(output_dir, masks, n_classes, kpt_cls)]

    # transform masks
    num_processes = 2
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(transform_mask, tasks)
#
#
#
# def files_newer_than(files, day, hour, mins):
#     delta = datetime.timedelta(days=day, hours=hour, minutes=mins)
#     now = datetime.datetime.now()
#     file_list = []
#     for a in files:
#         c_time = datetime.datetime.fromtimestamp(os.path.getctime(a))
#         if now - delta < c_time:
#             file_list.append(a)
#     return file_list
#
# # ======================================================================================================================
#
# for m in masks:
#     try:
#         transform_mask(
#             output_dir=output_dir,
#             path_to_mask=m, n_classes=6, kpt_cls=(5, 6),
#         )
#     except:
#         continue
#
# # ======================================================================================================================
#
#
# if __name__ == '__main__':
#
#     base_dir = "/home/anjonas/public/Public/Jonas/Data/ESWW007/SingleLeaf/Output"
#     # base_dir = "Z:/Public/Jonas/Data/ESWW007/SingleLeaf/Output"
#     masks = glob.glob(f'{base_dir}/*/mask2/*.png')
#
#     # only list files which have been successfully transformed
#     existing_output = glob.glob(f'{base_dir}/*/result/piecewise/*.JPG')
#     processed = glob.glob(f'{base_dir}/*/mask_aligned/piecewise/*.png')
#     # proc = files_newer_than(processed, day=2, hour=0, mins=0)
#     bnames = [os.path.basename(x).replace(".JPG", "") for x in existing_output]
#     pnames = [os.path.basename(x).replace(".png", "") for x in processed]
#     masks = [m for m in masks if os.path.basename(m).replace(".png", "") in bnames]
#     masks = [m for m in masks if os.path.basename(m).replace(".png", "") not in pnames]
#
#     num_processes = 2
#
#     print("processing " + str(len(masks)) + " samples")
#
#     # Create a multiprocessing pool to parallelize the loop
#     with multiprocessing.Pool(processes=num_processes) as pool:
#         pool.map(transform_mask, masks)

# ======================================================================================================================



