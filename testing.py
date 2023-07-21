# Rename images of single leaves using datetime of image capture from exif
# and QR code in images

# import the necessary packages
import imutils
import cv2
import glob
import os
import exifread
import pandas as pd
from pyzbar.pyzbar import decode

parent_dir = "Z:/Public/Jonas/Data/ESWW007/SingleLeaf"
jpeg_dirs = [os.path.join(parent_dir, d, "JPEG_cam") for d in os.listdir(parent_dir)]

for dir in jpeg_dirs[8:9]:
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
