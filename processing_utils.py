import glob
import numpy as np
import os
from pathlib import Path
import shutil
import pandas as pd
import re


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


def rearrange_images_plotwise(folder):
    walker = iter(os.walk(folder))
    root, dirs, files = next(walker)
    from_name_list = []
    to_name_list = []
    # remove any files that are not JPG images from the list of files
    files = [f for f in files if ".JPG" in f]
    for f in files:
        from_name = Path(root + "/" + f)
        # ensure that the file name has the expected name structure
        plot_name = f.split("_")[2]
        if not bool(re.match(plot_name[:7], "ESWW009")):
            print(F"unexpected name pattern: {plot_name} in {from_name}")
            break
        # make a folder per plot and move images
        plot_UID = os.path.basename(f).split("_")[2]
        plot_folder = Path(root + "/" + plot_UID)
        plot_folder.mkdir(parents=True, exist_ok=True)
        to_name = Path(plot_folder / f)
        shutil.move(src=from_name, dst=to_name)
        # record the old and current full file names for logging
        from_name_list.append(from_name)
        to_name_list.append(to_name)
    # make a log file of the files movements
    dict = {"File.Creation.Name": from_name_list, "File.Renaming.NewName": to_name_list}
    log = pd.DataFrame(dict)
    log.to_csv(f'{root}/moving_log_iter2.csv', index=False)


def check_plotwise(folder):
    # list all created subdirectories (plots)
    sub_root, sub_dirs, sub_files = next(os.walk(folder))
    # filter subdirectories that are not plots
    sub_dirs = [d for d in sub_dirs if "ESWW009" in d]
    # list all images and test if there is unique leaf numbers
    problem_folder_list = []
    for sd in sub_dirs:
        full_path = Path(sub_root + "/" + sd)
        print(full_path)
        files = glob.glob(f"{full_path}/*.JPG")
        f = [os.path.basename(x).split("_")[3].replace(".JPG", "") for x in files]
        # if leaf numbers are not unique record the folder for manual checking
        if len(f) != len(set(f)):
            problem_folder_list.append(full_path)
    # make a log file of the files movements
        dict = {"Directory": problem_folder_list}
    log = pd.DataFrame(dict)
    log.to_csv(f'{folder}/problem_log.csv', index=False)


def move_to_parent_dir(folder):
    # list all images in plot subdirectories
    imgs = glob.glob(f'{folder}/*/*.JPG')
    from_name_list = []
    to_name_list = []
    for i in imgs:
        base_name = os.path.basename(i)
        file_dir = os.path.dirname(i)
        parent_dir = os.path.split(file_dir)[0]
        from_name = i
        to_name = os.path.join(parent_dir, base_name)
        shutil.move(src=from_name, dst=to_name)
        # record the old and current full file names for logging
        from_name_list.append(from_name)
        to_name_list.append(to_name)
    # make a log file of the files movements
    dict = {"File.Creation.Name": from_name_list, "File.Renaming.NewName": to_name_list}
    log = pd.DataFrame(dict)
    log.to_csv(f'{parent_dir}/moving_back_log.csv', index=False)


# Function to check if a path contains any of the plot strings
def contains_any_plot(path, plots):
    return any(plot in path for plot in plots)


def rename_double(file_paths):
    from_name_list = []
    to_name_list = []
    for fp in file_paths:
        base_name = os.path.basename(fp)
        leaf_nr_new = int(base_name.split("_")[-1].replace(".JPG", "")) + 10
        new_base_name = "_".join(base_name.split("_")[:-1]) + "_" + str(leaf_nr_new) + ".JPG"
        from_name = fp
        to_name = os.path.dirname(fp) + "\\" + new_base_name
        os.rename(from_name, to_name)
        # record the old and current full file names for logging
        from_name_list.append(from_name)
        to_name_list.append(to_name)
    # make a log file of the files movements
    dict = {"File.Creation.Name": from_name_list, "File.Renaming.NewName": to_name_list}
    log = pd.DataFrame(dict)
    return log








