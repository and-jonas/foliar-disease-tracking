
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
from Test import Stitcher
import warnings
matplotlib.use('Qt5Agg')


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

    # crop
    crop = img[0:500, 2000:3000]

    imageio.imwrite(f"Z:/Public/Jonas/011_STB_leaf_tracking/Figures/guessing_images/{bn}", crop)

