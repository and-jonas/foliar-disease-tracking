# ======================================================================================================================
# Transforms key point annotations exported from CVAT in 'CVAT for images 1.1' format
# to the format required for ultralytics YOLOv8
# ======================================================================================================================

import pandas as pd
from bs4 import BeautifulSoup
import glob
import os

# load annotation file
with open('Z:/Public/Jonas/Data/ESWW007/CvatDatasets/Markers/Set4/annotations.xml', 'r') as f:
    data = f.read()

# Passing the stored data inside
# the beautifulsoup parser
Bs_data = BeautifulSoup(data, "xml")

# Finding all instances of tag `image`
b_unique = Bs_data.find_all('image')

# get list of all image names
image_names = glob.glob('Z:/Public/Jonas/Data/ESWW007/CvatDatasets/Markers/Set4/Images/*.JPG')
image_names = [os.path.basename(x) for x in image_names]

# iterate over all annotated images
for image_name in image_names:
    # extract attributes of the first instance of the tag
    b_name = Bs_data.find('image', {'name': image_name})
    value = b_name.find_all('points')
    v = [value[i].get('points') for i in range(len(value))]

    out = []
    for element in v:
        out.extend([x.split(",") for x in element.split(";")])

    result = []
    for p in out:
        print(p)
        cl = 0
        c_x = int(float(p[0]))/8192  # transform into relative coordinates
        c_y = int(float(p[1]))/5464
        s1 = 0.00800  # approximate size of a marker
        s2 = s1
        x = c_x
        y = c_y
        result.append([cl, c_x, c_y, s1, s2, x, y])

    # create output .txt file
    df = pd.DataFrame(data=result)
    txt_name = image_name.replace(".JPG", ".txt")
    df.to_csv(f"Z:/Public/Jonas/Data/ESWW007/CvatDatasets/Markers/Set4/{txt_name}",
              sep=" ",
              index=False,
              header=False)

# ======================================================================================================================
# split and copy to dataset directory
# ======================================================================================================================

import numpy as np
import random
from pathlib import Path
import shutil

images = glob.glob("Z:/Public/Jonas/Data/ESWW007/CvatDatasets/Markers/Set*/Images/*.JPG")
labels = glob.glob("Z:/Public/Jonas/Data/ESWW007/CvatDatasets/Markers/Set*/*.txt")

# ensure uniqueness of the images
img_files = [os.path.basename(x) for x in images]
lab_files = [os.path.basename(x) for x in labels]
unique_img_idx = np.unique(img_files, return_index=True)[1]
unique_label_idx = np.unique(lab_files, return_index=True)[1]
images = [images[x] for x in range(len(images)) if x in unique_img_idx]
labels = [labels[x] for x in range(len(labels)) if x in unique_label_idx]

random.seed(10)
validation_images = random.sample(images, 18)
train_images = [item for item in images if item not in validation_images]
random.seed(10)
validation_labels = random.sample(labels, 18)
train_labels = [item for item in labels if item not in validation_labels]
lst_key = ["validation"] * len(validation_images) +["train"] * len(train_images)

imgs = validation_images + train_images
labs = validation_labels + train_labels

data_to = "Z:/Public/Jonas/Data/ESWW007/CvatDatasets/Markers/datasets/markers"
data_to = "C:/Users/anjonas/PycharmProjects/STBLeaf/datasets/markers"

# copy images and masks
Path(f'{data_to}/images/validation').mkdir(exist_ok=True, parents=True)
Path(f'{data_to}/images/train').mkdir(exist_ok=True, parents=True)
Path(f'{data_to}/labels/validation').mkdir(exist_ok=True, parents=True)
Path(f'{data_to}/labels/train').mkdir(exist_ok=True, parents=True)
for i in range(len(lst_key)):
    print(i)
    base_name = os.path.basename(imgs[i])
    # shutil.copy(imgs[i], f'{data_to}/images/{lst_key[i]}/{base_name.replace(".JPG", ".jpg")}')
    shutil.copy(labs[i], f'{data_to}/labels/{lst_key[i]}/{base_name.replace(".JPG", ".txt")}')










