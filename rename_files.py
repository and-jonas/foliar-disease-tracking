# ======================================================================================================================
# Rename images of single leaves using datetime of image capture from exif
# and QR code in image
# Rename images where two image series exist for the same experimental plot
# ======================================================================================================================

# import the necessary packages
import imutils
import cv2
import glob
import os
import exifread
import pandas as pd
from pyzbar.pyzbar import decode

parent_dir = "Z:/Public/Jonas/Data/ESWW007/SingleLeaf"
# jpeg_dirs = [os.path.join(parent_dir, d, "JPEG_cam") for d in os.listdir(parent_dir)]
jpeg_dirs = ["Z:/Public/Jonas/Data/ESWW007/SingleLeaf/20230608/JPEG"]  # head-over images


for dir in jpeg_dirs:
    files_in_dir = glob.glob(f'{dir}/*.JPG')
    old_names = []
    new_names = []
    for file in files_in_dir:
        id = None
        print(file)
        img = cv2.imread(file)
        im = imutils.resize(img, width=2000)
        # read barcode
        barcode = decode(im)
        try:
            id = barcode[0].data.decode('utf-8')
        except IndexError:
            id = os.path.basename(file).split(".")[0]
        # get image acquisition date time
        with open(file, 'rb') as f:
            tags = exifread.process_file(f)
        dtime = tags['EXIF DateTimeOriginal'].values
        dtime = dtime.replace(":", "")
        dtime = dtime.replace(" ", "_")
        full_id = "_".join([dtime, id])
        new_name = "".join([full_id, ".JPG"])
        new_name = os.path.join(os.path.dirname(file), new_name)
        new_names.append(new_name)
        old_names.append(file)
        os.rename(file, new_name)
    # create renaming log
    file_list = [old_names, new_names]
    df = pd.DataFrame(file_list).transpose()
    df.columns = ['old_names', 'new_names']
    df.to_csv(f"{dir}/renaming_log.csv")

# rename where two series (cohorts) exist for the same plot
dates_cohort2 = os.listdir("Z:/Public/Jonas/Data/ESWW007/SingleLeaf")[17:31]
dates_cohort2.remove('20230613')

parent_dir = "Z:/Public/Jonas/Data/ESWW007/SingleLeaf"
jpeg_dirs = [os.path.join(parent_dir, d, "JPEG_cam") for d in dates_cohort2]
labels_dirs = [os.path.join(parent_dir, d, "JPEG_cam/runs/pose/predict*/labels") for d in dates_cohort2]

# for dir in jpeg_dirs:
for dir in labels_dirs[1:]:
    # files_in_dir = glob.glob(f'{dir}/*.JPG')
    files_in_dir = glob.glob(f'{dir}/*.txt')
    files_to_rename = [f for f in files_in_dir if "ESWW0070228" in f or "ESWW0070054" in f]
    if len(files_to_rename) > 20:
        print("whoa!! stopping.")
        break
    old_names = []
    new_names = []
    for file in files_to_rename:
        print(file)
        # img_nr = int(os.path.basename(file).split("_")[-1].replace(".JPG", ""))
        img_nr = int(os.path.basename(file).split("_")[-1].replace(".txt", ""))
        # new_img_nr = ".".join([str(img_nr + 10), "JPG"])
        new_img_nr = ".".join([str(img_nr + 10), "txt"])
        bb = "_".join(os.path.basename(file).split("_")[:-1])
        new_name = "_".join([bb, new_img_nr])
        new_name = os.path.join(os.path.dirname(file), new_name)
        new_names.append(new_name)
        old_names.append(file)
        os.rename(file, new_name)
    # create renaming log
    file_list = [old_names, new_names]
    df = pd.DataFrame(file_list).transpose()
    df.columns = ['old_names', 'new_names']
    # df.to_csv(f"{dir}/renaming_log_3.csv")
    df.to_csv(f"{os.path.dirname(file)}/renaming_log_3.csv")


