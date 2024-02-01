
from pathlib import Path
import numpy as np
import pandas as pd
import json
import cv2
from PIL import Image
import matplotlib
import imageio
import matplotlib.pyplot as plt
import utils
import glob
import os
import copy
import skimage
import pickle
from scipy.spatial import distance as dist
from ImageStitcher import Stitcher
import warnings
matplotlib.use('Qt5Agg')

from datetime import datetime

overlays = glob.glob('Z:/Public/Jonas/011_STB_leaf_tracking/output/ts/ESWW0070020_1/overlay/*.png')
orig_images = glob.glob('Z:/Public/Jonas/Data/ESWW007/SingleLeaf/Output/ESWW0070020_1/result/projective/*.JPG')

for o in orig_images:

    img = Image.open(o)
    img = np.array(img)

    bn = os.path.basename(o)

    scale_percent = 50  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

guessing_images = glob.glob('Z:/Public/Jonas/Data/ESWW007/SingleLeaf/Output/ESWW0070054_10/result/piecewise/*.JPG')
for gi in guessing_images:

    img = Image.open(gi)
    img = np.array(img)
    bn = os.path.basename(gi)
    dt = bn[0:15]
    datetime_object = datetime.strptime(dt, '%Y%m%d_%H%M%S')

    # crop
    crop = img[0:300, 2200:2800]

    # add date time to image
    crop = cv2.putText(crop, str(datetime_object), org=(20, 275),  fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                          fontScale=0.9, color=(255, 255, 255), thickness=2)

    # add grid to image
    grid_y = list(range(0, crop.shape[0], 50))
    for g_y in grid_y:
        cv2.line(crop, (0, g_y), (crop.shape[1], g_y), (0, 0, 255), 1, 1)
    grid_x = list(range(0, crop.shape[1], 100))
    for g_x in grid_x:
        cv2.line(crop, (g_x, 0), (g_x, crop.shape[0]), (0, 0, 255), 1, 1)

    plt.imshow(crop)

    imageio.imwrite(f"Z:/Public/Jonas/011_STB_leaf_tracking/Figures/guessing_images/{bn}", crop)

