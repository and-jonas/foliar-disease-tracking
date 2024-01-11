import glob
import os.path
import utils
import lesion_utils
import matplotlib
import matplotlib.pyplot as plt
import imageio
import numpy as np
from matplotlib import path
import pandas as pd
import cv2
import copy
from scipy import ndimage
from scipy.spatial.distance import cdist
from scipy import ndimage as ndi
matplotlib.use('Qt5Agg')
from pathlib import Path

# ======================================================================================================================
# Extract lesion data
# ======================================================================================================================

# Initialize unique object labels
next_label = 1
all_objects = {}
labels = {}

# Containment threshold for area overlap
contour_overlap_threshold = 0.3

workdir = 'Z:/Public/Jonas/Data/ESWW007/SingleLeaf/Output'
path_images = f'{workdir}/*/result/piecewise'
path_aligned_masks = f'{workdir}/*/mask_aligned/piecewise'
path_kpts = f'{workdir}/*/keypoints/'

frames = glob.glob(f'{path_aligned_masks}/*0020_7.png')
images = glob.glob(f'{path_images}/*0020_7.JPG')
num_frames = len(frames)

out_path = 'Z:/Public/Jonas/Data/ESWW007/SingleLeaf/Output/ESWW0070020_7'

# make output dirs

dirs = [Path(f'{out_path}/mask_pp/img_lesion_checker'), Path(f'{out_path}/Data/leaf'), Path(f'{out_path}/DATA/lesion'),
        Path(f'{out_path}/mask_pp/1'), Path(f'{out_path}/mask_pp/2'), Path(f'{out_path}/mask_pp/3')]
for d in dirs:
    d.mkdir(exist_ok=True, parents=True)

# Process each frame in the time series
# for frame_number in range(1, num_frames + 1):
for frame_number in range(1, 7):

    print("processing frame " + os.path.basename(frames[frame_number-1]))

    png_name = os.path.basename(frames[frame_number - 1])
    data_name = png_name.replace(".png", ".txt")
    sample_name = png_name.replace(".png", "")

    # ==================================================================================================================
    # 1. Pre-processing
    # ==================================================================================================================

    # Load the multi-class segmentation mask
    frame_ = cv2.imread(frames[frame_number-1], cv2.IMREAD_GRAYSCALE)

    # load key point coordinates
    if frame_number == 1:
        kpts = [(0, 0), (frame_.shape[1], 0), (frame_.shape[1], frame_.shape[0]), (0, frame_.shape[0])]
    else:
        kpts_fn = glob.glob(f'{path_kpts}/{sample_name}.txt')[0]
        kpts0 = pd.read_csv(kpts_fn)
        kpts = utils.make_point_list(np.asarray(kpts0))

    print(np.unique(frame_))

    # get leaf mask (without insect damage!)
    mask_leaf = np.where((frame_ >= 41) & (frame_ != 153), 1, 0).astype("uint8")
    mask_leaf = np.where(mask_leaf, 255, 0).astype("uint8")

    # get lesion mask
    frame = np.where(frame_ == 102, 255, 0).astype("uint8")
    frame = utils.filter_objects_size(mask=frame, size_th=500, dir="smaller")
    # fill small holes
    kernel = np.ones((3, 3), np.uint8)
    frame = cv2.morphologyEx(frame, cv2.MORPH_DILATE, kernel, iterations=2)
    frame = cv2.morphologyEx(frame, cv2.MORPH_ERODE, kernel, iterations=2)
    # remove some artifacts, e.g., around insect damage
    frame = cv2.morphologyEx(frame, cv2.MORPH_ERODE, kernel, iterations=1)
    frame = cv2.morphologyEx(frame, cv2.MORPH_DILATE, kernel, iterations=1)
    # reformat
    frame = np.where(frame, 255, 0).astype("uint8")

    # ==================================================================================================================
    # 2. Get leaf mask
    # ==================================================================================================================

    # Find the reference point (mean of x and y coordinates)
    ref_point = np.mean(kpts, axis=0)

    # Calculate polar angles and sort the points
    sorted_points = sorted(kpts, key=lambda p: np.arctan2(p[1] - ref_point[1], p[0] - ref_point[0]))

    # transform coordinates to a path
    grid_path = path.Path(sorted_points, closed=False)

    # create a mask of the image
    xcoords = np.arange(0, frame.shape[0])
    ycoords = np.arange(0, frame.shape[1])
    coords = np.transpose([np.repeat(ycoords, len(xcoords)), np.tile(xcoords, len(ycoords))])

    # Create mask
    leaf_mask = grid_path.contains_points(coords, radius=-0.5)
    leaf_mask = np.swapaxes(leaf_mask.reshape(frame.shape[1], frame.shape[0]), 0, 1)
    leaf_mask = np.where(leaf_mask, 1, 0).astype("uint8")
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_DILATE, kernel, iterations=2)
    # leaf_mask = np.where(leaf_mask, leaf_mask, np.nan)

    # reduce to roi delimtied by the key points
    leaf_checker = mask_leaf * leaf_mask
    cv2.imwrite(f'{out_path}/mask_pp/1/{png_name}', leaf_checker)

    # ==================================================================================================================
    # 2. Watershed segmentation for object separation
    # ==================================================================================================================

    # fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    # axs[0].imshow(frame)
    # axs[0].set_title('original')
    # axs[1].imshow(seg)
    # axs[1].set_title('transformed')
    # plt.show(block=True)

    if frame_number == 1:
        seg = frame
    else:
        seg_lag = seg
        if len(np.unique(markers)) > 1:
            seg = lesion_utils.get_object_watershed_labels(current_mask=frame, markers=markers)
        else:
            seg = frame

    # important to avoid small water shed segments that cannot be processed
    # this is probably because of small shifts in frames over time (imperfect alignment)
    # removes small false positives
    seg = utils.filter_objects_size(mask=seg, size_th=1000, dir="smaller")

    # multiply with leaf mask
    seg = seg * leaf_mask

    if frame_number > 2:
        seg = lesion_utils.complement_mask(leaf_mask=leaf_mask, seg_lag=seg_lag, seg=seg, kpts=kpts0)

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

    # check size again
    seg = utils.filter_objects_size(mask=seg, size_th=50, dir="smaller")

    # generate complete watershed markers
    _, markers, _, _ = cv2.connectedComponentsWithStats(seg, connectivity=8)
    cv2.imwrite(f'{out_path}/mask_pp/2/{png_name}', seg)

    # ==================================================================================================================
    # 4. Analyze each lesion: label and extract data
    # ==================================================================================================================

    # find contours
    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # if not lesions are found, the original image without overlay is saved
    if len(contours) < 1:
        image_with_labels = copy.copy(img) * np.stack([leaf_mask, leaf_mask, leaf_mask], axis=2)
        imageio.imwrite(f'{out_path}/mask_pp/img_lesion_checker/{png_name}', image_with_labels)
        print("skipping")
        continue

    # Process each detected object in the current frame
    checker = copy.copy(img)
    lesion_data = []
    for idx, contour in enumerate(contours):

        print("-" + str(idx))

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

        in_leaf_checker = np.unique(leaf_mask[np.where(roi)[0], np.where(roi)[1]])[0]

        # fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
        # axs[0].imshow(roi)
        # axs[0].set_title('original')
        # axs[1].imshow(frame_)
        # axs[1].set_title('transformed')
        # plt.show(block=True)
        #
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
            mask_leaf=leaf_checker,
            img=empty_img,
            checker=checker
        )

        if in_leaf_checker == 0:
            # collect output data
            lesion_data.append({'label': current_label,
                                'area': np.nan,
                                'perimeter': np.nan,
                                'solidity': np.nan,
                                'analyzable_perimeter': np.nan,
                                'edge_perimeter': np.nan,
                                'neigh_perimeter': np.nan,
                                'max_width': np.nan,
                                'max_height': np.nan,
                                'n_pycn': np.nan})
        else:

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
            _, _, w, h = x, y, w, h = cv2.boundingRect(contour)

            # extract pycnidia number
            pycn_mask = np.where(roi, frame_, 0)
            n_pycn = len(np.where(pycn_mask == 204)[0])

            # collect output data
            lesion_data.append({'label': current_label,
                                'area': contour_area,
                                'perimeter': contour_perimeter,
                                'solidity': contour_solidity,
                                'analyzable_perimeter': analyzable_perimeter,
                                'edge_perimeter': edge_perimeter,
                                'neigh_perimeter': neigh_perimeter,
                                'max_width': w,
                                'max_height': h,
                                'n_pycn': n_pycn})

    # Update the labels with the new matches
    labels = object_matches
    all_objects = objects
    # print("n_labels = " + str(len(labels)))

    # ==================================================================================================================
    # 5. Analyze leaf
    # ==================================================================================================================

    # summary stats
    la_tot = (frame_.shape[0] * frame_.shape[1]) - len(np.where(frame_ == 0)[0])  # roi area - background pixels
    la_damaged = len(np.where((frame_ != 0) & (frame_ != 51))[0])
    la_healthy = len(np.where(frame_ == 51)[0])
    la_damaged_f = la_damaged/la_tot
    la_healthy_f = la_healthy/la_tot
    la_insect = len(np.where(frame_ == 153)[0])
    n_pycn = len(np.where(frame_ == 204)[0])
    n_rust = len(np.where(frame_ == 255)[0])
    n_lesion = len(contours)
    placl = len(np.where((frame_ == 102) | (frame_ == 204))[0])/(la_tot - la_insect)
    pycn_density = n_pycn/(la_tot - la_insect)
    rust_density = n_rust/(la_tot - la_insect)

    # distribution metrics
    out = ndi.distance_transform_edt(np.bitwise_not(seg))
    out[frame_ == 0] = np.nan
    out[out == 0] = np.nan
    mean_dist = np.nanmean(out)
    std_dist = np.nanstd(out)
    cv_dist = std_dist/mean_dist
    n_comps, output, stats, centroids = cv2.connectedComponentsWithStats(seg, connectivity=8)
    distance = cdist(centroids[1:], centroids[1:], metric='euclidean')
    np.fill_diagonal(distance, np.nan)
    shortest_dist = np.nanmin(distance, axis=1)
    mean_shortest_dist = np.mean(shortest_dist)
    std_shortest_dist = np.std(shortest_dist)
    cv_shortest_dist = std_shortest_dist/mean_shortest_dist

    # grab data
    leaf_data = [
        {
            'la_tot': la_tot,
            'la_damaged': la_damaged,
            'la_healthy': la_healthy,
            'la_damaged_f': la_damaged_f,
            'la_healthy_f': la_healthy_f,
            'la_insect': la_insect,
            'n_pycn': n_pycn,
            'n_rust': n_rust,
            'n_lesion': n_lesion,
            'placl': placl,
            'pycn_density': pycn_density,
            'rust_density': rust_density,
            'mean_dist': mean_dist,
            'std_dist': std_dist,
            'cv_dist': cv_dist,
            'mean_shortest_dist': mean_shortest_dist,
            'std_shortest_dist': std_shortest_dist,
            'cv_shortest_dist': cv_shortest_dist,
        },
    ]

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(leaf_data)

    # Export the DataFrame to a CSV file
    df.to_csv(f'{out_path}/DATA/leaf/{data_name}', index=False)

    # ==================================================================================================================
    # 6. Create output
    # ==================================================================================================================

    # save lesion data
    result = pd.DataFrame(lesion_data, columns=lesion_data[0].keys())
    result.to_csv(f'{out_path}/DATA/lesion/{data_name}', index=False)

    # Draw and save the labeled objects on the frame
    frame_with_labels = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR) * np.stack([leaf_mask, leaf_mask, leaf_mask], axis=2)
    image_with_labels = copy.copy(out_checker) * np.stack([leaf_mask, leaf_mask, leaf_mask], axis=2)
    for label, (x, y, w, h) in labels.items():
        cv2.rectangle(frame_with_labels, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame_with_labels, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(image_with_labels, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image_with_labels, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imwrite(f'{out_path}/mask_pp/3/{png_name}', frame_with_labels)
    imageio.imwrite(f'{out_path}/mask_pp/img_lesion_checker/{png_name}', image_with_labels)

# ==================================================================================================================
