import glob
import numpy as np
import os


def get_series(paths_to_crops):
    """
    Creates two lists of file paths: to key point coordinate files and to images
    for each of the samples monitored over time, stored in date-wise folders.
    :return:
    """
    crop_series = []
    images = glob.glob(f'{paths_to_crops}/*.JPG')
    image_image_id = ["_".join(os.path.basename(l).split("_")[2:4]).replace(".JPG", "") for l in images]
    uniques = np.unique(image_image_id)

    for unique_sample in uniques:
        image_idx = [index for index, image_id in enumerate(image_image_id) if unique_sample == image_id]
        sample_image_names = [images[i] for i in image_idx]
        # sort to ensure sequential processing of subsequent images
        sample_image_names = sorted(sample_image_names, key=lambda i: os.path.splitext(os.path.basename(i))[0])
        crop_series.append(sample_image_names)

    return crop_series


def get_output_paths(path_output, series):

    sample_id = "_".join(os.path.basename(series[0]).split("_")[2:4]).replace(".JPG", "")
    # generate output paths for each sample and create directories
    sample_output_path = path_output / sample_id
    mask_path = sample_output_path / "mask"

    return mask_path
