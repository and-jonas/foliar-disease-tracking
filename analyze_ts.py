import glob
import os.path
import utils
import lesion_utils
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import numpy as np
import pandas as pd
import cv2
import copy
matplotlib.use('Qt5Agg')

# ======================================================================================================================
# Mask pre-processing
# ======================================================================================================================

path = "Z:/Public/Jonas/Data/ESWW007/SingleLeaf/Output/*/mask_aligned"

masks = glob.glob(f'{path}/*0020_1.png')
masks = sorted(masks)

for i, mask in enumerate(masks):

    print(mask)

    out_dir = os.path.dirname(os.path.dirname(mask))
    png_name = os.path.basename(mask)
    out_name = f'{out_dir}/mask_pp/1/{png_name}'

    m = Image.open(mask)
    m = np.asarray(m)
    lesions = np.where(m == 170, 255, 0).astype("uint8")
    lesions = utils.filter_objects_size(mask=lesions, size_th=100, dir="smaller")

    if i == 0:
        imageio.imwrite(out_name, lesions)
        lesions_lag = lesions
    else:
        lesions = np.bitwise_or(lesions, lesions_lag).astype("uint8")
        # lesions = lesions
        imageio.imwrite(out_name, lesions)
        lesions_lag = lesions

# ======================================================================================================================
# Watershed segmentation of merged lesion objects
# ======================================================================================================================

path = "Z:/Public/Jonas/Data/ESWW007/SingleLeaf/Output/*/mask_pp/1"
masks = glob.glob(f'{path}/*0020_1.png')
masks = sorted(masks)

SEG = []
for i, mask in enumerate(masks):

    out_dir = os.path.dirname(os.path.dirname(mask))
    png_name = os.path.basename(mask)
    out_name = f'{out_dir}/2/{png_name}'

    m = Image.open(mask)
    m = np.asarray(m).astype("uint8")

    # get connected component labels as markers for watershed segmentation
    if i == 0:
        seg = m
        _, markers, _, _ = cv2.connectedComponentsWithStats(m, connectivity=4)

    else:
        seg = lesion_utils.get_object_watershed_labels(current_mask=m, markers=markers)
        _, markers, _, _ = cv2.connectedComponentsWithStats(seg, connectivity=4)

    SEG.append(markers)
    imageio.imwrite(out_name, np.uint8(seg))

# ======================================================================================================================
# Extract lesion data
# ======================================================================================================================

# Initialize unique object labels
next_label = 1
labels = {}

path = "Z:/Public/Jonas/Data/ESWW007/SingleLeaf/Output"

frames = glob.glob(f'{out_dir}/2/*.png')
images = glob.glob(f'{os.path.dirname(out_dir)}/result/*.JPG')
num_frames = 14

# Process each frame in the time series
for frame_number in range(1, num_frames + 1):

    print(frame_number)

    # Load the binary segmentation mask for the current frame
    frame = cv2.imread(frames[frame_number-1], cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(images[frame_number-1], cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    kernel = np.ones((3, 3), np.uint8)  # Adjust the kernel size as needed

    png_name = os.path.basename(frames[frame_number - 1])
    data_name = png_name.replace(".png", ".txt")

    # Find contours in the current frame
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a dictionary to store object matches in the current frame
    object_matches = {}

    # Process each detected object in the current frame
    checker = copy.copy(img)
    data = []
    for contour in contours:

        x, y, w, h = map(int, cv2.boundingRect(contour))
        rect = cv2.boundingRect(contour)

        # Shrink bounding box
        w_ = int(w * 0.5)
        h_ = int(h * 0.3)
        x_ = int(x + (w - w_) / 2)
        y_ = int(y + (h - h_) / 2)

        # Check if the object's bounding box intersects with any previous object's bounding box
        is_new_object = True
        for label, (prev_x, prev_y, prev_w, prev_h) in labels.items():
            if (
                x_ < prev_x + prev_w and
                x_ + w_ > prev_x and
                y_ < prev_y + prev_h and
                y_ + h_ > prev_y
            ):
                is_new_object = False
                object_matches[label] = (x, y, w, h)
                break

        # If the object does not intersect with any previous object, assign a new label
        if is_new_object:
            object_matches[next_label] = (x, y, w, h)
            next_label += 1

        # change format of the bounding rectangle
        rect = lesion_utils.get_bounding_boxes(rect=rect)

        # extract roi
        empty_mask_all, empty_mask, empty_img, ctr_obj = lesion_utils.select_roi(rect=rect, img=img, mask=frame)

        # extract RGB profiles
        prof, out_checker, spl, spl_points = lesion_utils.spline_contours(
            mask_obj=empty_mask,
            mask_all=empty_mask_all,
            img=empty_img,
            checker=checker
        )

        # extract analyzable perimeter
        analyzable_perimeter = len(spl[1])/len(spl[0])

        # extract other contour properties
        contour_area = cv2.contourArea(contour)
        contour_perimeter = cv2.arcLength(contour, True)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_solidity = float(contour_area) / hull_area

        # collect output data
        data.append({'lesion_id': label,
                     'area': contour_area,
                     'perimeter': contour_perimeter,
                     'solidity': contour_solidity,
                     'analyzable_perimeter': analyzable_perimeter})

    # save checker
    imageio.imwrite(f'{out_dir}/lesion_checker/{png_name}', out_checker[0])

    # save lesion data
    result = pd.DataFrame(data, columns=data[0].keys())
    result.to_csv(f'{os.path.dirname(out_dir)}/DATA/{data_name}', index=False)

    del data

    # Update the labels with the new matches
    labels = object_matches

    # Draw and save the labeled objects on the frame
    frame_with_labels = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for label, (x, y, w, h) in labels.items():
        cv2.rectangle(frame_with_labels, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame_with_labels, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imwrite(f'{out_dir}/3/{png_name}', frame_with_labels)

