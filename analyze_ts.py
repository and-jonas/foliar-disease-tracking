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
from scipy import ndimage
matplotlib.use('Qt5Agg')

# ======================================================================================================================
# Extract lesion data
# ======================================================================================================================

# Initialize unique object labels
next_label = 1
all_objects = {}
labels = {}

# Containment threshold for area overlap
contour_overlap_threshold = 0.3
# area_threshold = 1000
# centroid_distance_threshold = 20  # Adjust the centroid distance threshold as needed

workdir = 'Z:/Public/Jonas/Data/ESWW007/SingleLeaf/Output'

path_images = f'{workdir}/*/result'
path_aligned_masks = f'{workdir}/*/mask_aligned'

frames = glob.glob(f'{path_aligned_masks}/*0020_1.png')
images = glob.glob(f'{path_images}/*0020_1.JPG')
num_frames = len(frames)

path = 'Z:/Public/Jonas/Data/ESWW007/SingleLeaf/Output/ESWW0070020_1'

# Process each frame in the time series
for frame_number in range(1, num_frames + 1):

    print("processing frame " + os.path.basename(frames[frame_number-1]))

    png_name = os.path.basename(frames[frame_number - 1])
    data_name = png_name.replace(".png", ".txt")

    # ==================================================================================================================
    # 1. Pre-processing
    # ==================================================================================================================

    # Load the multi-class segmentation mask
    frame = cv2.imread(frames[frame_number-1], cv2.IMREAD_GRAYSCALE)

    # get leaf mask
    mask_leaf = np.where(frame >= 127, 1, 0).astype("uint8")
    mask_leaf = ndimage.binary_fill_holes(mask_leaf)
    mask_leaf = np.where(mask_leaf, 255, 0).astype("uint8")

    # get lesion mask
    frame = np.where(frame == 191, 255, 0).astype("uint8")
    frame = utils.filter_objects_size(mask=frame, size_th=150, dir="smaller")
    frame = ndimage.binary_fill_holes(frame)
    frame = np.where(frame, 255, 0).astype("uint8")

    # ==================================================================================================================
    # 2. Watershed segmentation for object separation
    # ==================================================================================================================

    if frame_number == 1:
        seg = frame
        _, markers, _, _ = cv2.connectedComponentsWithStats(frame, connectivity=8)
    else:
        seg_lag = seg
        if len(np.unique(markers)) > 1:
            seg = lesion_utils.get_object_watershed_labels(current_mask=frame, markers=markers)
        else:
            seg = frame
        _, markers, _, _ = cv2.connectedComponentsWithStats(seg, connectivity=8)

    # ==================================================================================================================
    # 3. Identify and add undetected lesions from previous frame
    # ==================================================================================================================

    # get image
    img = cv2.imread(images[frame_number-1], cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # init dict
    object_matches = {}  # matches
    objects = {}  # all objects

    # check if there are any missing objects from the last frame in the current frame
    # update the mask if needed
    for lab, (lag_x, lag_y, lag_w, lag_h) in all_objects.items():
        out1 = seg_lag[lag_y:lag_y + lag_h, lag_x:lag_x + lag_w]
        out2 = seg[lag_y:lag_y + lag_h, lag_x:lag_x + lag_w]
        overlap = np.sum(np.bitwise_and(out1, out2)) / (255*len(np.where(out1)[1]))
        # if the object cannot be retrieved in the current mask,
        # paste the object from the previous frame into the current one
        if overlap < 0.1:
            seg[lag_y:lag_y + lag_h, lag_x:lag_x + lag_w] = seg_lag[lag_y:lag_y + lag_h, lag_x:lag_x + lag_w]
    cv2.imwrite(f'{path}/mask_pp/2/{png_name}', seg)

    # ==================================================================================================================
    # 4. Analyze each lesion: label and extract data
    # ==================================================================================================================

    # find contours
    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # if not lesions are found, the original image without overlay is saved
    if len(contours) < 1:
        imageio.imwrite(f'{path}/mask_pp/img_lesion_checker/{png_name}', img)
        print("skipping")
        continue

    # Process each detected object in the current frame
    checker = copy.copy(img)
    data = []
    for idx, contour in enumerate(contours):

        # print("-" + str(idx))

        x, y, w, h = map(int, cv2.boundingRect(contour))
        objects[idx] = (x, y, w, h)
        rect = cv2.boundingRect(contour)

        # Shrink bounding box
        w_ = int(w * 0.5)
        h_ = int(h * 0.2)
        x_ = int(x + (w - w_) / 2)
        y_ = int(y + (h - h_) / 2)

        # Calculate the centroid of the current object
        current_centroid = np.array([x_ + w_ / 2, y_ + h_ / 2])

        roi = lesion_utils.select_roi_2(rect=rect, mask=seg)
        # plt.imshow(roi)
        # plt.show(block=True)

        is_new_object = True
        for lag_label, (lag_x, lag_y, lag_w, lag_h) in labels.items():

            # print("--" + str(lag_label))

            rect_lag = (lag_x, lag_y, lag_w, lag_h)

            # Calculate the area of the larger bounding box
            area_smaller = lag_w * lag_h

            # Calculate the intersection area
            intersection_area = max(0, min(x + w, lag_x + lag_w) - max(x, lag_x)) * \
                                max(0, min(y + h, lag_y + lag_h) - max(y, lag_y))

            # Calculate the area overlap as a ratio of the smaller bounding box area
            area_overlap = intersection_area / area_smaller

            # isolate the objects
            roi_lag = lesion_utils.select_roi_2(rect=rect_lag, mask=seg_lag)

            # fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
            # axs[0].imshow(roi)
            # axs[0].set_title('curr')
            # axs[1].imshow(roi_lag)
            # axs[1].set_title('is')
            # plt.show(block=True)

            # get areas and overlap
            current_area = np.sum(np.logical_and(roi, roi))
            lag_area = np.sum(np.logical_and(roi_lag, roi_lag))
            int_area = np.sum(np.logical_and(roi, roi_lag))
            contour_overlap = int_area / lag_area

            # get centroid of previous object
            ctr_lag = np.array([lag_x + lag_w / 2, lag_y + lag_h / 2])

            # calculate the distance between centroids
            ctr_dist = np.linalg.norm(current_centroid - ctr_lag)

            if contour_overlap >= contour_overlap_threshold:
                is_new_object = False
                object_matches[lag_label] = (x, y, w, h)
                current_label = lag_label  # Update the label to the existing object's label
                break

        # If the object is not significantly overlapped with any previous object, assign a new label
        if is_new_object:
            object_matches[next_label] = (x, y, w, h)
            current_label = next_label  # Update the label to the newly assigned label
            next_label += 1

        # change format of the bounding rectangle
        rect = lesion_utils.get_bounding_boxes(rect=rect)

        # extract roi
        empty_mask_all, _, empty_img, ctr_obj = lesion_utils.select_roi(rect=rect, img=img, mask=seg)

        # extract RGB profile, checker image, spline normals, and spline base points
        prof, out_checker, spl, spl_points = lesion_utils.spline_contours(
            mask_obj=roi,
            mask_all=empty_mask_all,
            mask_leaf=mask_leaf,
            img=empty_img,
            checker=checker
        )

        # extract perimeter lengths
        analyzable_perimeter = len(spl[1])/len(spl[0])
        edge_perimeter = len(spl[3])/len(spl[0])
        neigh_perimeter = len(spl[2])/len(spl[0])

        # extract other lesion properties
        # these are extracted from the original (un-smoothed) contour
        contour_area = cv2.contourArea(contour)
        contour_perimeter = cv2.arcLength(contour, True)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_solidity = float(contour_area) / hull_area

        # collect output data
        data.append({'label': current_label,
                     'area': contour_area,
                     'perimeter': contour_perimeter,
                     'solidity': contour_solidity,
                     'analyzable_perimeter': analyzable_perimeter,
                     'edge_perimeter': edge_perimeter,
                     'neigh_perimeter': neigh_perimeter})

    # Update the labels with the new matches
    labels = object_matches
    all_objects = objects
    # print("n_labels = " + str(len(labels)))

    # ==================================================================================================================
    # 5. Create output
    # ==================================================================================================================

    # save lesion data
    result = pd.DataFrame(data, columns=data[0].keys())
    result.to_csv(f'{path}/DATA/{data_name}', index=False)

    # Draw and save the labeled objects on the frame
    frame_with_labels = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)
    image_with_labels = copy.copy(out_checker)
    for label, (x, y, w, h) in labels.items():
        cv2.rectangle(frame_with_labels, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame_with_labels, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(image_with_labels, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image_with_labels, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imwrite(f'{path}/mask_pp/3/{png_name}', frame_with_labels)
    imageio.imwrite(f'{path}/mask_pp/img_lesion_checker/{png_name}', image_with_labels)

# ==================================================================================================================
