
import pathlib
import cv2
from tqdm import tqdm
import json
import numpy as np
import glob

DATA_SRC = '/home/anjonas/public/Public/Jonas/Data/ESWW007/SingleLeaf/2023*/JPEG_cam/'
IMG_EXT = '.JPG'
LABEL_EXT = '.json'
DATA_DST = '/home/anjonas/public/Public/Jonas/011_STB_leaf_tracking/data/single_leaves_jonas_export/crops'
DST_EXT = '.jpg'
IMG_H = 2048

data_dir = pathlib.Path(DATA_SRC)

paths = glob.glob(f'{DATA_SRC}/*{IMG_EXT}')

diff = []
missing_files = []

for img_path in tqdm(paths):
    label_path = str(img_path).replace(IMG_EXT, LABEL_EXT)
    if not pathlib.Path(label_path).exists():
        print("File: {} does not exist".format(label_path))
        missing_files.append(str(label_path))
        continue

    # Read in the json file.
    with open(label_path, 'r') as label_file:
        # Load JSON data from the file
        data = json.load(label_file)
    # Parse bb coordinates
    bb = np.array(data["bounding_box"])
    diff.append(bb[0, 1] - bb[-1, 1])

    # Calculate the centroid
    mw, mh = np.mean(bb, axis=0)
    # Read in image
    img = cv2.imread(str(img_path))

    # Crop According to the centroid
    h_min = int(mh) - int(IMG_H/2)

    if h_min < 0:
        print("Bounding Box outside of the image")
        continue

    if h_min + IMG_H >= len(img):
        print("Bounding Box outside of the image")
        continue

    img_cropped = img[h_min:h_min + IMG_H, :, :]

    # Save the cropped image
    export_path = pathlib.Path(DATA_DST)
    export_path = img_path.stem + DST_EXT

    export_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(export_path), img_cropped)

print('done')

