
# ======================================================================================================================
# Script to align the ROIs from images in a series
# ======================================================================================================================

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import utils
import glob
import os
from pathlib import Path
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

l_series = label_series[31]
i_series = image_series[31]

for l_series, i_series in zip(label_series[7:15], image_series[7:15]):

    if len(l_series) != len(i_series):
        print("series not of equal length!")
        print(l_series)
        print(i_series)
        break

    roi_widths = []
    for j in range(len(l_series)):

        image_id = os.path.basename(l_series[j]).replace(".txt", "")
        sample_id = "_".join(os.path.basename(l_series[j]).split("_")[2:4]).replace(".txt", "")

        print(str(sample_id) + ": image " + str(j))

        # create output paths for each sample
        sample_output_path = path_output / sample_id
        overlay_path = sample_output_path / "overlay"
        overlay2_path = sample_output_path / "overlay2"
        crops_path = sample_output_path / "crops"
        result_path = sample_output_path / "result"
        for p in (overlay_path, overlay2_path, crops_path, result_path):
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
        outliers_x = utils.reject_outliers(x)
        outliers_y = utils.reject_outliers(y)
        outliers = outliers_x + outliers_y
        point_list = np.delete(point_list, outliers, 0)
        rect = cv2.minAreaRect(point_list)

        # TODO enlarge bounding box to capture additional lesions outside of the tagged range
        # expand bounding box to the edge of the image
        (center, (w, h), angle) = rect
        # if w > h:
        #     w = w + 2000
        # else:
        #     h = h + 2000
        # rect = (center, (w, h), angle)
        if angle > 45:
            angle = angle - 90

        # rotate the image about its center
        rows, cols = img.shape[0], img.shape[1]
        M_img = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        img_rot = cv2.warpAffine(img, M_img, (cols, rows))

        # rotate the bounding box about its center
        M_box = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        box = cv2.boxPoints(rect)
        pts = np.intp(cv2.transform(np.array([box]), M_box))[0]
        pts[pts < 0] = 0

        # order bounding box points clockwise
        pts = utils.order_points(pts)

        # pts = utils.expand_bbox_to_image_edge(pts, img=img_rot)

        # crop the roi from the rotated image
        img_crop = img_rot[pts[0][1]:pts[2][1], pts[0][0]:pts[1][0]]

        # log the width of the roi to detect outliers in series
        roi_widths.append(img_crop.shape[1])

        if j == 0:
            cv2.imwrite(f'{crops_path}/{image_id}.png', cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB),
                        [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else:
            # log the width of the roi to detect outliers in series
            size_outliers = utils.reject_size_outliers(roi_widths, max_diff=100)
            if not size_outliers:
                # log the width of the roi to detect outliers in series
                cv2.imwrite(f'{crops_path}/{image_id}.png', cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB),
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])
            else:
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

        # draw key points on overlay image as check
        for i, point in enumerate(kpts):
            cv2.circle(img_crop, (point[0], point[1]), radius=7, color=(0, 0, 255), thickness=-1)
            cv2.putText(img_crop,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        org=point,
                        text=str(i),
                        thickness=4,
                        fontScale=2,
                        color=(255, 0, 0))
        if j != 0:
            for i, point in enumerate(kpts_ref):
                cv2.circle(img_crop, (point[0], point[1]), radius=7, color=(255, 0, 0), thickness=-1)
        cv2.imwrite(f'{overlay_path}/{image_id}.JPG', cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))

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
                prev_image_id = os.path.basename(l_series[j - 1]).replace(".txt", "")
                previous_image = Image.open(f'{result_path}/{prev_image_id}.JPG')
                previous_image = np.asarray(previous_image)
                current_image = save_img

                # adjust size by padding if needed; images must have equal height for stitching
                (w1, h1, _) = previous_image.shape
                (w2, h2, _) = current_image.shape
                if w2 > w1:
                    previous_image = cv2.copyMakeBorder(previous_image, 0, w2 - w1, 0, 0, cv2.BORDER_CONSTANT)
                elif w1 > w2:
                    current_image = cv2.copyMakeBorder(current_image, 0, w1 - w2, 0, 0, cv2.BORDER_CONSTANT)

                scale_factor = 1
                width = int(img_crop.shape[1] * scale_factor)
                height = int(img_crop.shape[0] * scale_factor)
                dim = (width, height)
                prev = cv2.resize(previous_image, dim, interpolation=cv2.INTER_LINEAR)
                curr = cv2.resize(current_image, dim, interpolation=cv2.INTER_LINEAR)

                stitcher = Stitcher()
                try:
                    (result, vis, H) = stitcher.stitch(images=[copy.copy(prev), copy.copy(curr)],
                                                       masks=[None, None],
                                                       showMatches=True)
                except TypeError:
                    "could not match images"
                    continue
                #
                # t, th, sc, sh = utils.getComponents(H)
                # T = np.array([
                #     [1, 0, -t[0]],
                #     [0, 1, 0]
                # ], dtype=np.float32)
                #
                # warped = cv2.warpAffine(current_image, T, (previous_image.shape[1], previous_image.shape[0]))
                #
                # fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
                # axs[0].imshow(previous_image)
                # axs[0].set_title('original')
                # axs[1].imshow(warped)
                # axs[1].set_title('transformed')
                # plt.show(block=True)

                # dim_small = np.float32([[0, 0], [prev.shape[0], 0], [prev.shape[0], prev.shape[1]], [0, prev.shape[1]]])
                # dim_large = np.float32([[0, 0], [previous_image.shape[0], 0], [previous_image.shape[0], previous_image.shape[1]], [0, previous_image.shape[1]]])
                # H_scale = cv2.getPerspectiveTransform(src=dim_small,
                #                                       dst=dim_large)
                # H_tot = np.dot(np.linalg.inv(H), np.linalg.inv(H_scale))
                # warped = cv2.warpPerspective(current_image, H_tot,
                #                              (previous_image.shape[1], previous_image.shape[0]))
                # warped = skimage.util.img_as_ubyte(warped)
                # plt.imshow(warped)

                # warp image by applying the inverse of the homography matrix
                warped = cv2.warpPerspective(current_image, np.linalg.inv(H),
                                             (previous_image.shape[1], previous_image.shape[0]))
                warped = skimage.util.img_as_ubyte(warped)
                # cv2.imwrite(f'{result_path}/{image_id}.JPG', cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

                # warp key points by applying the inverse of the homography matrix
                kpts_warped = [utils.warp_point(x[0], x[1], np.linalg.inv(H)) for x in kpts]
                # kpts_warped = np.intp(cv2.transform(np.array([kpts]), T))[0]

                # draw overlay
                for i, point in enumerate(kpts_warped):
                    cv2.circle(warped, (point[0], point[1]), radius=7, color=(0, 0, 255), thickness=-1)
                    cv2.putText(warped,
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                org=point,
                                text=str(i),
                                thickness=4,
                                fontScale=2,
                                color=(255, 0, 0))
                for i, point in enumerate(kpts):
                    cv2.circle(warped, (point[0], point[1]), radius=7, color=(0, 255, 0), thickness=-1)
                cv2.imwrite(f'{overlay2_path}/{image_id}.JPG', cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

                # match key points with those on the first image of the series
                # by searching for the closest points
                # if non is found in proximity, eliminate from both images
                src, dst = utils.find_keypoint_matches(kpts_warped, kpts, kpts_ref)

            # perspective Transform with the first image of the series as destination
            tform = skimage.transform.ProjectiveTransform()
            # tform = skimage.transform.PolynomialTransform()
            # tform = skimage.transform.PiecewiseAffineTransform()
            tform.estimate(src, dst)
            warped = skimage.transform.warp(save_img, tform)
            warped = skimage.util.img_as_ubyte(warped)
            cv2.imwrite(f'{result_path}/{image_id}.JPG', cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            del size_outliers
