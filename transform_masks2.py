
# ======================================================================================================================
# Verifies that the extracted image rotation matrix, bounding box coordinates,
# and final transformation matrix are correct.
# Crops the roi, rotates and translates roi, transforms detected points,
# and outputs the rectified masks for further processing.
# ======================================================================================================================

from PIL import Image
import numpy as np
import cv2
import skimage
import pickle
import imageio
import glob
from pathlib import Path
import os
import json
import utils
import multiprocessing
from importlib import reload

# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('Qt5Agg')

dir_img = "/home/anjonas/public/Public/Radek/01_Data/A_Eschikon_Field_Experiments/single_leaves_jonas"
dir_mask = "/home/anjonas/public/Public/Jonas/011_STB_leaf_tracking/data/single_leaves_jonas_export/predictions"
dir_meta = "/home/anjonas/public/Public/Jonas/Data/ESWW007/SingleLeaf/Output"
# dir_img = "Z:/Public/Radek/01_Data/A_Eschikon_Field_Experiments/single_leaves_jonas"
# dir_mask = "Z:/Public/Jonas/011_STB_leaf_tracking/data/single_leaves_jonas_export/predictions"
# dir_meta = "Z:/Public/Jonas/Data/ESWW007/SingleLeaf/Output"
masks = glob.glob(f'{dir_mask}/*.png')
masks = sorted(masks)

existing_output = glob.glob(f'{dir_meta}/*/result/piecewise/*.JPG')
bnames = [os.path.basename(x).replace(".JPG", "") for x in existing_output]
masks = [m for m in masks if os.path.basename(m).replace(".png", "") in bnames]

# ======================================================================================================================


# Process masks
def transform_mask(path_to_mask):

    print(path_to_mask)

    # get names
    base_name = os.path.basename(path_to_mask)
    jpg_name = base_name.replace(".png", ".JPG")
    stem_name = base_name.replace(".png", "")
    leaf_name = "_".join(stem_name.split("_")[2:4])

    # get mask
    mask = Image.open(path_to_mask)
    mask = np.asarray(mask)

    # get image
    image = f'{dir_img}/{jpg_name}'
    img = Image.open(image)
    img = np.asarray(img)

    # ==================================================================================================================

    # get the target (warped roi)
    target = Image.open(f"{dir_meta}/{leaf_name}/result/piecewise/{stem_name}.JPG")
    target = np.asarray(target)

    # get bounding box localization info
    output_p = f"{dir_meta}/{leaf_name}/roi/{stem_name}.json"
    f = open(output_p)
    data = json.load(f)
    rot = np.asarray(data['rotation_matrix'])
    bbox = np.asarray(data['bounding_box'])
    box_ = np.intp(bbox)

    tform_piecewise = None
    tform_projective = None
    if "transformation_matrix" in [k for k in data.keys()]:
        tform_projective = np.asarray(data['transformation_matrix'])
    f.close()
    try:
        with open(f'{dir_meta}/{leaf_name}/roi/{stem_name}_tform_piecewise.pkl', 'rb') as file:
            tform_piecewise = pickle.load(file)
    except FileNotFoundError:
        pass

    # get image roi
    full_img = np.zeros((5464, 8192, 3)).astype("uint8")
    mw, mh = map(int, np.mean(box_, axis=0))
    full_img[mh-1024:mh+1024, :] = img
    rows, cols = full_img.shape[0], full_img.shape[1]
    full_img_rot = cv2.warpAffine(full_img, rot, (cols, rows))
    ll = full_img_rot[box_[0][1]:box_[2][1], box_[0][0]:box_[1][0]]

    # full mask
    full_mask = np.zeros((5464, 8192)).astype("uint8")
    full_mask[mh - 1024:mh + 1024, :] = mask

    # ==================================================================================================================

    # get coordinates of pycnidia and rust pustules
    pycn = np.where(full_mask == 5)
    rust = np.where(full_mask == 6)

    # remove points from the mask (would not be maintained during transformation)
    full_mask = utils.remove_points_from_mask(mask=full_mask, points=pycn)
    full_mask = utils.remove_points_from_mask(mask=full_mask, points=rust)

    # rotate mask
    full_mask_rot = cv2.warpAffine(full_mask, rot, (cols, rows))

    # extract points
    pycn_point_list = np.array([[a, b] for a, b in zip(pycn[1], pycn[0])], dtype=np.int32)
    rust_point_list = np.array([[a, b] for a, b in zip(rust[1], rust[0])], dtype=np.int32)

    # transform points and filter for roi
    if len(pycn_point_list) != 0:
        pycn_trf = utils.rotate_translate_warp_points(
            points=pycn_point_list,
            rot=rot, box=box_,
            mat_proj=tform_projective, mat_pw=tform_piecewise,
            target_shape=target.shape
        )
    else:
        pycn_trf = [None, None]
    if len(rust_point_list) != 0:
        rust_trf = utils.rotate_translate_warp_points(
            points=rust_point_list,
            rot=rot, box=box_,
            mat_proj=tform_projective, mat_pw=tform_piecewise,
            target_shape=target.shape
        )
    else:
        rust_trf = [None, None]
    # ==================================================================================================================

    # shrink mask to actual roi
    leaf_mask = full_mask_rot[box_[0][1]:box_[2][1], box_[0][0]:box_[1][0]]

    out_dir = Path(f'{dir_meta}/{leaf_name}/mask_aligned/')
    out_dir_pw = out_dir / "piecewise"
    out_dir_proj = out_dir / "projective"
    for dir in [out_dir_pw, out_dir_proj]:
        dir.mkdir(exist_ok=True, parents=True)

    tform = [tform_projective, tform_piecewise]
    out_dir = [out_dir_proj, out_dir_pw]

    for i, tf in enumerate(tform):

        mask_name = f'{out_dir[i]}/{base_name}'

        if tf is not None:

            # warp mask
            lm = np.stack([leaf_mask, leaf_mask, leaf_mask], axis=2)
            warped_m = skimage.transform.warp(lm, tf, output_shape=target.shape)
            warped_m = skimage.util.img_as_ubyte(warped_m[:, :, 0])

            warped_m = utils.add_points_to_mask(
                mask=warped_m,
                pycn_trf=pycn_trf[i],
                rust_trf=rust_trf[i]
            )

            imageio.imwrite(mask_name, warped_m)

            # # warp image
            # warped = skimage.transform.warp(ll, tf, output_shape=target.shape)
            # warped = skimage.util.img_as_ubyte(warped)

        else:

            leaf_mask = utils.add_points_to_mask(
                mask=leaf_mask,
                pycn_trf=pycn_trf[i],
                rust_trf=rust_trf[i]
            )

            imageio.imwrite(mask_name, leaf_mask)


# ======================================================================================================================


for m in masks:
    transform_mask(m)

# ======================================================================================================================


if __name__ == '__main__':

    dir_img = "/home/anjonas/public/Public/Radek/01_Data/A_Eschikon_Field_Experiments/single_leaves_jonas"
    dir_mask = "/home/anjonas/public/Public/Jonas/011_STB_leaf_tracking/data/single_leaves_jonas_export/predictions"
    dir_meta = "/home/anjonas/public/Public/Jonas/Data/ESWW007/SingleLeaf/Output"
    masks = glob.glob(f'{dir_mask}/*.png')

    existing_output = glob.glob(f'{dir_meta}/*/result/piecewise/*.JPG')
    bnames = [os.path.basename(x).replace(".JPG", "") for x in existing_output]
    masks = [m for m in masks if os.path.basename(m).replace(".png", "") in bnames]

    # num_processes = multiprocessing.cpu_count()  # Use the number of CPU cores
    num_processes = 20

    # Create a multiprocessing pool to parallelize the loop
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(transform_mask, masks)

# ======================================================================================================================
