
# imports
from pathlib import Path
import matplotlib.image as mpimg
import pandas as pd
import re
import random
import os
from evaluation_utils import capture_evaluation_positions_GUI
import processing_utils

workdir = "Z:/Public/Jonas/Data/ESWW007/SingleLeaf/Output"
trs_type = "piecewise"
path_to_registered = f'{workdir}/*/result/{trs_type}'
output_path = "Z:/Public/Jonas/011_STB_leaf_tracking/validation"

series = processing_utils.get_series(path_to_registered)

for s in series:

    # randomly select a pair of images from the series
    path_img1, path_img2 = random.sample(s, k=2)
    pattern = r'[/\\]'
    sample_id = re.split(pattern, path_img1)[7]

    # load required files
    try:
        img1 = mpimg.imread(path_img1)
        img2 = mpimg.imread(path_img2)
    except FileNotFoundError:
        print("Image not found. Skipping.")
        continue

    # sample evaluation positions
    # close GUI if proposed image is inadequate
    # terminate console to stop early
    try:
        coords1 = []
        coords2 = []
        coords1, coords2 = capture_evaluation_positions_GUI(
            img1=img1,
            img2=img2,
            coords_img1=coords1,
            coords_img2=coords2
        )
        if not coords1:
            raise ValueError('image pair not used, loading next')
    except ValueError:
        print('image pair not used, loading next')
        continue

    # add image path to positions
    for cs1, cs2 in zip(coords1, coords2):
        cs1.update({"img_id": os.path.basename(path_img1), "sample_id": sample_id, "trs_type": trs_type})
        cs2.update({"img_id": os.path.basename(path_img2), "sample_id": sample_id, "trs_type": trs_type})

    # save training coordinates to .csv
    eval1 = pd.DataFrame(coords1)
    eval2 = pd.DataFrame(coords2)
    if not Path(output_path).exists():
        Path(output_path).mkdir(parents=True, exist_ok=True)
    filename1 = f'{output_path}/{sample_id}_1.csv'
    filename2 = f'{output_path}/{sample_id}_2.csv'
    eval1.to_csv(filename1, index=False)
    eval2.to_csv(filename2, index=False)

