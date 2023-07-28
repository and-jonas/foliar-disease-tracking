
# ======================================================================================================================
# Script to align the ROIs from images in a series
# ======================================================================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import json
import cv2
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import utils
import glob
import os
import copy
import skimage
from Test import Stitcher
import warnings
matplotlib.use('Qt5Agg')

# path_labels = "Z:/Public/Jonas/Data/ESWW007/SingleLeaf/Test/runs/pose/predict4/labels"
# path_images = "Z:/Public/Jonas/Data/ESWW007/SingleLeaf/Test"

path_labels = "Z:/Public/Jonas/Data/ESWW007/SingleLeaf/*/JPEG_cam/runs/pose/predict*/labels"
path_images = "Z:/Public/Jonas/Data/ESWW007/SingleLeaf/*/JPEG_cam"


def list_by_sample(path_labels, path_images):

    label_series = []
    image_series = []

    labels = glob.glob(f'{path_labels}/*.txt')
    images = glob.glob(f'{path_images}/*.JPG')
    label_image_id = ["_".join(os.path.basename(l).split("_")[2:4]).replace(".txt", "") for l in labels]
    image_image_id = ["_".join(os.path.basename(l).split("_")[2:4]).replace(".JPG", "") for l in images]
    uniques = np.unique(label_image_id)

    len(images)
    len(labels)

    if len(images) != len(labels):
        # raise Exception("list of images and list of coordinate files are not of equal length.")
        warnings.warn("list of images and list of coordinate files are not of equal length."
                      "Ignoring extra coordinate files.")

    print("found " + str(len(uniques)) + " unique sample names")

    for unique_sample in uniques:
        image_idx = [index for index, image_id in enumerate(image_image_id) if unique_sample == image_id]
        label_idx = [index for index, label_id in enumerate(label_image_id) if unique_sample == label_id]
        sample_image_names = [images[i] for i in image_idx]
        sample_labels = [labels[i] for i in label_idx]
        label_series.append(sample_labels)
        image_series.append(sample_image_names)

    return label_series, image_series

label_series, image_series = list_by_sample(
    path_labels=path_labels,
    path_images=path_images
)

path_output = Path("Z:/Public/Jonas/Data/ESWW007/SingleLeaf/Output")

l_series = label_series[26]
i_series = image_series[26]

with open(f"{path_output}/unmatched.txt", 'w') as file:
    pass

for l_series, i_series in zip(label_series[7:15], image_series[7:15]):

    if len(l_series) != len(i_series):
        print("series not of equal length!")
        print(l_series)
        print(i_series)
        break

    roi_widths = []
    for j in range(len(l_series)):
    # for j in range(8):

        image_id = os.path.basename(l_series[j]).replace(".txt", "")
        sample_id = "_".join(os.path.basename(l_series[j]).split("_")[2:4]).replace(".txt", "")

        print(str(sample_id) + ": image " + str(j))

        # generate output paths for each sample and create directories
        sample_output_path = path_output / sample_id
        overlay_path = sample_output_path / "overlay"
        roi_path = sample_output_path / "roi"
        result_path = sample_output_path / "result"
        for p in (overlay_path, roi_path, result_path):
            p.mkdir(parents=True, exist_ok=True)

        # get key point coordinates from YOLO output
        coords = pd.read_table(l_series[j], header=None, sep=" ")
        x = coords.iloc[:, 5] * 8192
        y = coords.iloc[:, 6] * 5464

        # get image
        img = Image.open(i_series[j])
        img = np.array(img)

        # remove outliers in the key point detections from YOLO errors,
        # get minimum area rectangle around retained key points
        point_list = np.array([[a, b] for a, b in zip(x, y)], dtype=np.int32)
        outliers_x = utils.reject_outliers(x, m=3.)  # larger extension, larger variation
        outliers_y = utils.reject_outliers(y, m=2.)  # smaller extension, smaller variation
        outliers = outliers_x + outliers_y
        point_list = np.delete(point_list, outliers, 0)
        rect = cv2.minAreaRect(point_list)

        # TODO enlarge bounding box to capture additional lesions outside of the tagged range
        (center, (w, h), angle) = rect
        if angle > 45:
            angle = angle - 90

        # rotate the image about its center
        rows, cols = img.shape[0], img.shape[1]
        M_img = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        img_rot = cv2.warpAffine(img, M_img, (cols, rows))

        # rotate the bounding box the image's center
        M_box = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        box = cv2.boxPoints(rect)
        pts = np.intp(cv2.transform(np.array([box]), M_box))[0]
        pts[pts < 0] = 0

        # order bounding box points clockwise
        pts = utils.order_points(pts)

        # record roi localization
        roi_loc = {'rotation_matrix': M_img.tolist(), 'bounding_box': pts.tolist()}

        # draw key points and bounding box on overlay image as check
        overlay = copy.copy(img)
        for point in point_list:
            cv2.circle(overlay, (point[0], point[1]), radius=15, color=(0, 0, 255), thickness=9)
        box_ = np.intp(box)
        cv2.drawContours(overlay, [box_], 0, (255, 0, 0), 9)
        overlay = cv2.resize(overlay, (0, 0), fx=0.25, fy=0.25)
        cv2.imwrite(f'{overlay_path}/{image_id}.JPG', cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

        # pts = utils.expand_bbox_to_image_edge(pts, img=img_rot)

        # crop the roi from the rotated image
        img_crop = img_rot[pts[0][1]:pts[2][1], pts[0][0]:pts[1][0]]

        # log the width of the roi to detect outliers in series
        roi_widths.append(img_crop.shape[1])
        if j == 0:
            init_roi_height = img_crop.shape[0]

        # detect size outliers and remove from the logged bbox width values
        if j != 0:
            size_outliers = utils.reject_size_outliers(roi_widths, max_diff=100)
            if size_outliers:
                del roi_widths[-1]

        # copy for later use
        save_img = copy.copy(img_crop)

        # rotate and translate key point coordinates
        kpts = np.intp(cv2.transform(np.array([point_list]), M_img))[0]

        # get tx and ty values for key point translation
        tx, ty = (-pts[0][0], -pts[0][1])
        translation_matrix = np.array([
            [1, 0, tx],
            [0, 1, ty]
        ], dtype=np.float32)

        # apply  translation to key points
        kpts = np.intp(cv2.transform(np.array([kpts]), translation_matrix))[0]

        # match all images in the series to the first image where possible
        if j == 0:
            kpts_ref = kpts
            cv2.imwrite(f'{result_path}/{image_id}.JPG', cv2.cvtColor(save_img, cv2.COLOR_BGR2RGB))
        elif j > 0:
            # match key points with those on the first image of the series
            # by searching for the closest points
            # if non is found in proximity, eliminate from both images
            src, dst = utils.find_keypoint_matches(kpts, kpts, kpts_ref)

            # if there are few matches, or if there is a different size from the expected,
            # there is likely a translation due to key point detection errors
            # try to fix by matching with the previous image in the series using SIFT features
            if len(src) < 12 or size_outliers:
                if len(src) < 12:
                    print("Key point mis-match. Matching on last image in series.")
                if size_outliers:
                    print("Size outlier detected. Matching on last image in series.")
                try:
                    prev_image_id = os.path.basename(l_series[j - 1]).replace(".txt", "")
                    previous_image = Image.open(f'{result_path}/{prev_image_id}.JPG')
                except FileNotFoundError:
                    prev_image_id = os.path.basename(l_series[j - 2]).replace(".txt", "")
                    previous_image = Image.open(f'{result_path}/{prev_image_id}.JPG')
                except FileNotFoundError:
                    continue
                previous_image = np.asarray(previous_image)
                current_image = save_img

                # adjust size by padding if needed; images must have equal height for stitching
                (w1, h1, _) = previous_image.shape
                (w2, h2, _) = current_image.shape
                if w2 > w1:
                    previous_image = cv2.copyMakeBorder(previous_image, 0, w2 - w1, 0, 0, cv2.BORDER_CONSTANT)
                elif w1 > w2:
                    current_image = cv2.copyMakeBorder(current_image, 0, w1 - w2, 0, 0, cv2.BORDER_CONSTANT)

                # try stitching images using SIFT and RANSAC
                stitcher = Stitcher()
                try:
                    (result, vis, H) = stitcher.stitch(images=[copy.copy(previous_image), copy.copy(current_image)],
                                                       masks=[None, None],
                                                       showMatches=True)
                except TypeError:
                    print("could not match images")
                    f = open(f"{path_output}/unmatched.txt", 'a')
                    f.writelines(image_id + "\n")
                    f.close()
                    continue

                # warp image by applying the inverse of the homography matrix
                warped = cv2.warpPerspective(current_image, np.linalg.inv(H),
                                             (previous_image.shape[1], previous_image.shape[0]))
                warped = skimage.util.img_as_ubyte(warped)

                # warp key points by applying the inverse of the homography matrix
                kpts_warped = [utils.warp_point(x[0], x[1], np.linalg.inv(H)) for x in kpts]

                # try again to match key points with those on the first image of the series
                src, dst = utils.find_keypoint_matches(kpts_warped, kpts, kpts_ref)

            # perspective Transform with the first image of the series as destination
            tform = skimage.transform.ProjectiveTransform()
            # tform = skimage.transform.PolynomialTransform()
            # tform = skimage.transform.PiecewiseAffineTransform()
            try:
                tform.estimate(src, dst)
            except ValueError:
                print("could not derive transformation matrix")
                f = open(f"{path_output}/unmatched.txt", 'a')
                f.write(image_id)
                f.close()
                continue

            warped = skimage.transform.warp(save_img, tform, output_shape=(init_roi_height, roi_widths[0]))
            warped = skimage.util.img_as_ubyte(warped)
            cv2.imwrite(f'{result_path}/{image_id}.JPG', cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            # if os.path.getsize(f'{result_path}/{image_id}.JPG') < 700:
            #     print("could not derive transformation matrix")
            #     f = open(f"{path_output}/unmatched.txt", 'a')
            #     f.write(image_id)
            #     f.close()

            del size_outliers

            # add transformation matrix to the roi localization info
            roi_loc['transformation_matrix'] = tform.params.tolist()

        with open(f'{roi_path}/{image_id}.json', 'w') as outfile:
            json.dump(roi_loc, outfile)
