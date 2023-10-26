import cv2

import utils
from PIL import Image
import glob
import os
import numpy as np
import json
import imageio

# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('Qt5Agg')

import multiprocessing


dir_img = "/home/anjonas/public/Public/Radek/01_Data/A_Eschikon_Field_Experiments/single_leaves_jonas"
dir_meta = "/home/anjonas/public/Public/Jonas/Data/ESWW007/SingleLeaf/Output"
masks = glob.glob("Z/home/anjonas/public/Public/Jonas/011_STB_leaf_tracking/data/single_leaves_jonas_export/predictions/*.png")


# Define a function to process each mask
def process_mask(m):

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
        image = f'{dir_img}/{jpg_name}'
        img = Image.open(image)
    except FileNotFoundError:
        try:
            image = f'{dir_img}/{jpg_name}'
            img = Image.open(image)
        except FileNotFoundError:
            image = f'{dir_img}/{jpg_name}'
            img = Image.open(image)
    img = np.asarray(img)

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
    # rotate the image about the point that was the center in the original image
    full_img = np.zeros((5464, 8192, 3)).astype("uint8")

    # Calculate the centroid
    mw, mh = map(int, np.mean(box_, axis=0))

    full_img[mh-1024:mh+1024, :] = img
    full_mask = np.zeros((5464, 8192)).astype("uint8")
    full_mask[mh - 1024:mh + 1024, :] = mask
    full_mask_bin = np.where(full_mask == 2, 255, 0).astype("uint8")

    contours, _ = cv2.findContours(full_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        cv2.drawContours(full_img, c, -1, (255, 0, 0), 1)

    # overlay = utils.make_overlay(patch=full_img, mask=full_mask_bin, colors=[(0, 0, 1, 0.2)])
    # plt.imshow(full_img)
    imageio.imwrite(f'{dir_out}/{base_name}', full_img)

    # Rest of your code within the loop


if __name__ == '__main__':
    dir_img = "/home/anjonas/public/Public/Radek/01_Data/A_Eschikon_Field_Experiments/single_leaves_jonas"
    dir_meta = "/home/anjonas/public/Public/Jonas/Data/ESWW007/SingleLeaf/Output"
    masks = glob.glob("/home/anjonas/public/Public/Jonas/011_STB_leaf_tracking/data/single_leaves_jonas_export/predictions/*.png")
    dir_out = "/home/anjonas/public/Public/Jonas/Data/ESWW007/SingleLeaf/temp/overlay"

    # num_processes = multiprocessing.cpu_count()  # Use the number of CPU cores
    num_processes = 12

    # Create a multiprocessing pool to parallelize the loop
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_mask, masks)

# for m in masks:
#
#     print(m)
#
#     # get names
#     base_name = os.path.basename(m)
#     jpg_name = base_name.replace(".png", ".JPG")
#     stem_name = base_name.replace(".png", "")
#     leaf_name = "_".join(stem_name.split("_")[2:4])
#     date = stem_name.split("_")[0]
#
#     # get mask
#     mask = Image.open(m)
#     mask = np.asarray(mask)
#
#     # get image
#     try:
#         image = f'{dir_img}/{jpg_name}'
#         img = Image.open(image)
#     except FileNotFoundError:
#         try:
#             image = f'{dir_img}/{jpg_name}'
#             img = Image.open(image)
#         except FileNotFoundError:
#             image = f'{dir_img}/{jpg_name}'
#             img = Image.open(image)
#     img = np.asarray(img)
#
#     # get bounding box localization info
#     output_p = f"{dir_meta}/{leaf_name}/roi/{stem_name}.json"
#     f = open(output_p)
#     data = json.load(f)
#     rot = np.asarray(data['rotation_matrix'])
#     bbox = np.asarray(data['bounding_box'])
#     box_ = np.intp(bbox)
#
#     tform = None
#     if "transformation_matrix" in [k for k in data.keys()]:
#         tform = np.asarray(data['transformation_matrix'])
#     f.close()
#
#     # get roi
#     # rotate the image about the point that was the center in the original image
#     full_img = np.zeros((5464, 8192, 3)).astype("uint8")
#
#     # Calculate the centroid
#     mw, mh = map(int, np.mean(box_, axis=0))
#
#     full_img[mh-1024:mh+1024, :] = img
#     full_mask = np.zeros((5464, 8192)).astype("uint8")
#     full_mask[mh - 1024:mh + 1024, :] = mask
#     full_mask_bin = np.where(full_mask == 2, 255, 0).astype("uint8")
#
#     contours, _ = cv2.findContours(full_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     for c in contours:
#         cv2.drawContours(full_img, c, -1, (255, 0, 0), 1)
#
#     # overlay = utils.make_overlay(patch=full_img, mask=full_mask_bin, colors=[(0, 0, 1, 0.2)])
#     # plt.imshow(full_img)
#     imageio.imwrite(f'Z:/Public/Jonas/Data/ESWW007/SingleLeaf/temp/overlay/{base_name}', full_img)



