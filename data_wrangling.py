
import os
from pathlib import Path
import processing_utils
import glob
import pandas as pd
import imageio
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
from PIL import Image
import shutil

# dir_images = "Z:/Public/Jonas/Data/ESWW009/SingleLeaf"
dir_images = "/home/anjonas/public/Public/Jonas/Data/ESWW009/SingleLeaf"

# ======================================================================================================================
# STEP 1: Find duplicate images
# ======================================================================================================================

# rearrange images according to experimental plots
root, dirs, files = next(os.walk(dir_images))
for d in dirs:
    full_path = Path(root + "/" + d + "/JPEG_cam")
    print(full_path)
    processing_utils.rearrange_images_plotwise(folder=full_path)

# identify plots with multiple images for one or several leaves
# rearrange images according to experimental plots
for d in dirs:
    full_path = Path(root + "/" + d + "/JPEG_cam")
    processing_utils.check_plotwise(folder=full_path)

# ======================================================================================================================
# STEP 2: Rename where multiple series per plot were made (Fl0-1 and Fl0)
# ======================================================================================================================

# ==> plots 23, 57, 84, 94, 200 were measured in cohort 1 and cohort 2, temporally overlapping
# rename cohort 2 images from 1...10 to 11...20
# 1) find cohort 2 images for dates where both cohorts were imaged
from_double = glob.glob(f'{dir_images}/*/JPEG_cam/*/s2/*.JPG')
# 2) find cohort 2 images from plots included in cohort 1 in all successive dates
plots = ["ESWW0090023", "ESWW0090057", "ESWW0090084", "ESWW0090094", "ESWW0090200"]
relevant_dates = glob.glob(f'{dir_images}/*/JPEG_cam')[12:]
relevant_plots_in_dates1 = []
for rd in relevant_dates:
    all_imgs = glob.glob(f"{rd}/*/*.JPG")
    sel_imgs = [path for path in all_imgs if processing_utils.contains_any_plot(path, plots)]
    relevant_plots_in_dates1.extend(sel_imgs)

# ==> plots 132, 214 were measured in cohort 1 and cohort 3, not overlapping
# rename cohort 3 images from 1...10 to 11...20
plots = ["ESWW0090132", "ESWW0090214"]
relevant_dates = glob.glob(f'{dir_images}/*/JPEG_cam')[13:]
relevant_plots_in_dates2 = []
for rd in relevant_dates:
    all_imgs = glob.glob(f"{rd}/*/*.JPG")
    sel_imgs = [path for path in all_imgs if processing_utils.contains_any_plot(path, plots)]
    relevant_plots_in_dates2.extend(sel_imgs)

# combine all and rename
all = from_double + relevant_plots_in_dates1 + relevant_plots_in_dates2
log = processing_utils.rename_double(file_paths=all)
log.to_csv(f'{dir_images}/renaming_double_cohort_log.csv', index=False)

# ======================================================================================================================
# STEP 3: Move back to date-wise directories
# ======================================================================================================================

# move all images back to their  date-wise directory
root, dirs, files = next(os.walk(dir_images))
for d in dirs:
    full_path = Path(root + "/" + d + "/JPEG_cam")
    print(full_path)
    processing_utils.move_to_parent_dir(folder=full_path)

# ======================================================================================================================
# STEP 4: Rename mis-labelled images (mismatch between label and leaf)
# ======================================================================================================================

# load correction instructions
correction_info = pd.read_csv(f'{dir_images}/Output/Mismatched_leaf_MP.csv')

# get list of all images
all_imgs = glob.glob(f'{dir_images}/*/JPEG_cam/*.JPG')

# get images that must be rotated
img_mislabel = correction_info["mislabeled"].tolist()
img_mislabel = [mis for mis in img_mislabel if str(mis) != "nan"]
img_correct_label = correction_info["correct_label"].tolist()
img_correct_label = [cor for cor in img_correct_label if str(cor) != "nan"]

from_name_list = []
to_name_list = []
for i in range(len(img_mislabel)):
    img_mislabel_full = [img for img in all_imgs if img_mislabel[i] in img]
    from_name = img_mislabel_full[0]
    dirname = os.path.dirname(from_name)
    to_name = f'{dirname}/{img_correct_label[i]}.JPG'
    from_name_list.append(from_name)
    to_name_list.append(to_name)
    os.rename(from_name, to_name)
# make a log file of the files movements
dict = {"File.Old.Name": from_name_list, "File.New.Name": to_name_list}
log = pd.DataFrame(dict)
log.to_csv(f'{dir_images}/renaming_log.csv', index=False)

# ======================================================================================================================
# STEP 5:  Check again for duplicate images
# ======================================================================================================================

# rearrange images according to experimental plots
root, dirs, files = next(os.walk(dir_images))
for d in dirs:
    full_path = Path(root + "/" + d + "/JPEG_cam")
    print(full_path)
    processing_utils.rearrange_images_plotwise(folder=full_path)

# identify plots with multiple images for one or several leaves
# rearrange images according to experimental plots
for d in dirs:
    full_path = Path(root + "/" + d + "/JPEG_cam")
    processing_utils.check_plotwise(folder=full_path)

# move all images back to their  date-wise directory
root, dirs, files = next(os.walk(dir_images))
for d in dirs:
    full_path = Path(root + "/" + d + "/JPEG_cam")
    print(full_path)
    processing_utils.move_to_parent_dir(folder=full_path)

# delete obsolete folders
root, dirs, files = next(os.walk(dir_images))
for d in dirs:
    full_path = Path(root + "/" + d + "/JPEG_cam")
    print(full_path)
    r, old_folders, _ = next(os.walk(full_path))
    for of in old_folders:
        folder = f'{r}/{of}'
        if os.listdir(folder) == []:
            os.rmdir(folder)
        elif of == "runs":
            shutil.rmtree(folder)
        else:
            break

# ======================================================================================================================

# 2) ROTATE

# load correction instructions
correction_info = pd.read_csv(f'{dir_images}/Output/Mismatched_leaf_MP.csv')

# get list of all images
all_imgs = glob.glob(f'{dir_images}/*/JPEG_cam/*.JPG')

# get images that must be rotated
img_misoriented = correction_info["misoriented"].tolist()

for i in range(len(img_misoriented)):
    img_misoriented_full = [img for img in all_imgs if img_misoriented[i] in img]
    image = Image.open(img_misoriented_full[0])
    print(image.format)  # Should output 'JPEG'
    rotated_image = image.transpose(Image.ROTATE_180)
    rotated_image.save(img_misoriented_full[0], quality=100)

# ======================================================================================================================

# clean-up
dir_images = "Z:/Public/Jonas/Data/ESWW009/SingleLeaf"

# delete obsolete folders
root, dirs, files = next(os.walk(dir_images))
dirs = [d for d in dirs if "2024" in d]
for d in dirs:
    full_path = Path(root + "/" + d + "/JPEG_cam")
    print(full_path)
    r, old_folders, _ = next(os.walk(full_path))
    for of in old_folders:
        folder = f'{r}/{of}'
        if of == "runs":
            shutil.rmtree(folder)
            print("deleting runs")


# ======================================================================================================================
# X) COLLECTING PARTIAL OUTPUTS
# ======================================================================================================================

files = glob.glob("/home/anjonas/public/Public/Jonas/Data/ESWW009/SingleLeaf/Output/*/overlay/*.JPG")

for f in files:
    from_name = f
    base_name = os.path.basename(f)
    to_name = f'/home/anjonas/public/Public/Jonas/Data/ESWW009/SingleLeaf/Check/{base_name}'
    shutil.copy(src=from_name, dst=to_name)


# find missing outputs

_, dirs, _ = next(os.walk("/home/anjonas/public/Public/Jonas/Data/ESWW009/SingleLeaf/Output/"))
proc_files = ["_".join(os.path.basename(f).split("_")[2:4]).replace(".JPG", "") for f in files]


main_list = list(set(dirs) - set(proc_files))
