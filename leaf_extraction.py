
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
import pickle
from scipy.spatial import distance as dist
from Test import Stitcher
import warnings
matplotlib.use('Qt5Agg')

# path_labels = "Z:/Public/Jonas/Data/ESWW007/SingleLeaf/Test/runs/pose/predict4/labels"
# path_images = "Z:/Public/Jonas/Data/ESWW007/SingleLeaf/Test"

path_labels = "Z:/Public/Jonas/Data/ESWW007/SingleLeaf/*/JPEG_cam/runs/pose/predict2/labels"
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

l_series = label_series[3]
i_series = image_series[3]

with open(f"{path_output}/unmatched.txt", 'w') as file:
    pass

for l_series, i_series in zip(label_series, image_series):

    if len(l_series) != len(i_series):
        print("series not of equal length!")
        print(l_series)
        print(i_series)
        break

    roi_widths = []
    for j in range(len(l_series)):
    # for j in range(9):

        image_id = os.path.basename(l_series[j]).replace(".txt", "")
        sample_id = "_".join(os.path.basename(l_series[j]).split("_")[2:4]).replace(".txt", "")

        print(str(sample_id) + ": image " + str(j))

        # generate output paths for each sample and create directories
        sample_output_path = path_output / sample_id
        kpts_path = sample_output_path / "keypoints"
        overlay_path = sample_output_path / "overlay"
        roi_path = sample_output_path / "roi"
        result_path = sample_output_path / "result"
        result_pw = result_path / "piecewise"
        result_poly = result_path / "polynomial"
        result_proj = result_path / "projective"
        for p in (kpts_path, overlay_path, roi_path, result_pw, result_poly, result_proj):
            p.mkdir(parents=True, exist_ok=True)

        # get key point coordinates from YOLO output
        coords = pd.read_table(l_series[j], header=None, sep=" ")
        x = coords.iloc[:, 5] * 8192
        y = coords.iloc[:, 6] * 5464

        # get image
        img = Image.open(i_series[j])
        img = np.array(img)

        # remove double detections
        point_list = np.array([[a, b] for a, b in zip(x, y)], dtype=np.int32)
        dmat = dist.cdist(point_list, point_list, "euclidean")
        np.fill_diagonal(dmat, np.nan)
        dbl_idx = np.where(dmat < 100)[0].tolist()[::2]
        point_list = np.delete(point_list, dbl_idx, axis=0)
        x = np.delete(x, dbl_idx, axis=0)
        y = np.delete(y, dbl_idx, axis=0)

        # remove outliers in the key point detections from YOLO errors,
        # get minimum area rectangle around retained key points
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

        # rotate the bounding box about the image's center
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
            cv2.circle(overlay, (point[0], point[1]), radius=15, color=(0, 0, 255), thickness=2)
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

        # apply translation to key points
        kpts = np.intp(cv2.transform(np.array([kpts]), translation_matrix))[0]

        # # write rotated and translated key points to file for eventual later use
        # data = {"x": kpts[:, 0], "y": kpts[:, 1]}
        # df = pd.DataFrame(data)
        # df.to_csv(f'{kpts_path}/{image_id}.txt', index=False)

        # match all images in the series to the first image where possible
        if j == 0:
            kpts_ref = kpts
            cv2.imwrite(f'{result_pw}/{image_id}.JPG', cv2.cvtColor(save_img, cv2.COLOR_BGR2RGB))
            cv2.imwrite(f'{result_proj}/{image_id}.JPG', cv2.cvtColor(save_img, cv2.COLOR_BGR2RGB))
        elif j > 0:
            # match key points with those on the first image of the series
            # by searching for the closest points
            # if non is found in proximity, eliminate from both images
            src, dst = utils.find_keypoint_matches(current=kpts, current_orig=kpts, ref=kpts_ref)

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
                    previous_image = Image.open(f'{result_pw}/{prev_image_id}.JPG')
                except FileNotFoundError:
                    prev_image_id = os.path.basename(l_series[j - 2]).replace(".txt", "")
                    previous_image = Image.open(f'{result_pw}/{prev_image_id}.JPG')
                # except FileNotFoundError:
                #     continue
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
                    # continue

                # warp image by applying the inverse of the homography matrix
                warped = cv2.warpPerspective(current_image, np.linalg.inv(H),
                                             (previous_image.shape[1], previous_image.shape[0]))
                warped = skimage.util.img_as_ubyte(warped)

                # warp key points by applying the inverse of the homography matrix
                kpts_warped = [utils.warp_point(x[0], x[1], np.linalg.inv(H)) for x in kpts]

                # try again to match key points with those on the first image of the series
                src, dst = utils.find_keypoint_matches(kpts_warped, kpts, kpts_ref)

                kpts = np.asarray(kpts_warped)

            # # Try to fill in the missing corner markers based on distances in the previous frame
            #
            # # get left-most and right-most point and compare
            # prev_image_id = os.path.basename(l_series[j - 1]).replace(".txt", "")
            # kpts_lag = pd.read_csv(f'{kpts_path}/{prev_image_id}.txt')
            # tl_lag, tr_lag, br_lag, bl_lag = utils.order_points(np.asarray(kpts_lag))
            #
            # # get corners in reference image
            # tl_ref, tr_ref, bl_ref, br_ref = utils.get_corners(np.asarray(kpts_ref))
            #
            # # get corners of current image
            # tl_cur, tr_cur, bl_cur, br_cur = utils.get_corners(kpts)
            #
            # # get corners of dst
            # tl_dst, tr_dst, bl_dst, br_dst = utils.get_corners(np.asarray(dst))
            #
            # dist_tl = np.abs(tl_ref[0] - tl_cur[0])
            # dist_bl = np.abs(bl_ref[0] - bl_cur[0])
            # dist_tr = np.abs(tr_ref[0] - tr_cur[0])
            # dist_br = np.abs(br_ref[0] - br_cur[0])
            #
            # if any([dist_tl > 250, dist_bl > 250, dist_tr > 250, dist_br > 250]):
            #     try:
            #         prev_image_id = os.path.basename(l_series[j - 1]).replace(".txt", "")
            #         previous_image = Image.open(f'{result_path}/{prev_image_id}.JPG')
            #     except FileNotFoundError:
            #         prev_image_id = os.path.basename(l_series[j - 2]).replace(".txt", "")
            #         previous_image = Image.open(f'{result_path}/{prev_image_id}.JPG')
            #     except FileNotFoundError:
            #         continue
            #
            #     previous_image = np.asarray(previous_image)
            #     # plt.imshow(previous_image)
            #
            #     coords = pd.read_csv(f'{kpts_path}/{prev_image_id}.txt')
            #     x = coords["x"]
            #     y = coords["y"]
            #
            #     point_list_lag = np.array([[a, b] for a, b in zip(x, y)], dtype=np.int32)
            #     upper_lag = point_list_lag[point_list_lag[:, 1] < 300]
            #     lower_lag = point_list_lag[point_list_lag[:, 1] > 300]
            #
            #     # top left OK
            #     if dist_tl > 250:
            #         # in lag, find marker in top row that is closest to the current left-most marker
            #         anchor1 = upper_lag[np.argmin(dist.cdist(np.array([tl_dst]), upper_lag, "euclidean")[0])]
            #         # measure how far this marker is away from the left-most marker in the lag image
            #         dist1 = dist.cdist(np.array([anchor1]), np.array([tl_lag]), "euclidean")[0][0]
            #         # get distance between tl and bl
            #         dist2 = dist.cdist(np.array([bl_lag]), np.array([tl_lag]), "euclidean")[0][0]
            #         result = utils.Intersect2Circles(tl_dst, dist1, bl_dst, dist2)
            #         index = dist.cdist(np.array([[0, 0]]), np.array(result), "euclidean").argmin()
            #         try:
            #             index = dist.cdist(np.array([[0, 0]]), np.array(result), "euclidean").argmin()
            #             result = result[index]
            #             tl_inferred = np.asarray(result, dtype="int64").tolist()
            #             dst += [tl_inferred]
            #             src += [tl_ref.tolist()]
            #         except IndexError:
            #             pass
            #
            #     # bottom right OK
            #     if dist_br > 200:
            #         # in lag, find marker in bottom row that is closest to the current right-most marker
            #         anchor1 = lower_lag[np.argmin(dist.cdist(np.array([br_dst]), lower_lag, "euclidean")[0])]
            #         # measure how far this marker is away from the right-most marker in the lag image
            #         dist1 = dist.cdist(np.array([anchor1]), np.array([br_lag]), "euclidean")[0][0]
            #         # get distance between tr and br
            #         dist2 = dist.cdist(np.array([tr_lag]), np.array([br_lag]), "euclidean")[0][0]
            #         result = utils.Intersect2Circles(br_dst, dist1, tr_dst, dist2)
            #         try:
            #             # identify the point, based on distance to a rough proxy
            #             index = dist.cdist(np.array([[save_img.shape[1], save_img.shape[0]]]), np.array(result),
            #                                "euclidean").argmin()
            #             result = result[index]
            #             br_inferred = np.asarray(result, dtype="int64").tolist()
            #             dst += [br_inferred]
            #             src += [br_ref.tolist()]
            #         except IndexError:
            #             pass
            #
            #     # top right OK
            #     if dist_tr > 200:
            #         # in lag, find marker in top row that is closest to the current right-most marker
            #         anchor1 = upper_lag[np.argmin(dist.cdist(np.array([tr_dst]), upper_lag, "euclidean")[0])]
            #         # measure how far this marker is away from the right-most marker in the lag image
            #         dist1 = dist.cdist(np.array([anchor1]), np.array([tr_lag]), "euclidean")[0][0]
            #         # get distance between tr and br
            #         dist2 = dist.cdist(np.array([br_lag]), np.array([tr_lag]), "euclidean")[0][0]
            #         result = utils.Intersect2Circles(tr_dst, dist1, br_dst, dist2)
            #         try:
            #             # identify the point, based on distance to a rough proxy
            #             index = dist.cdist(np.array([[save_img.shape[1], 0]]), np.array(result), "euclidean").argmin()
            #             result = result[index]
            #             tr_inferred = np.asarray(result, dtype="int64").tolist()
            #             # tr_inferred = [4400, 0]
            #             dst += [tr_inferred]
            #             src += [tr_ref.tolist()]
            #         except (IndexError, ValueError):
            #             pass
            #
            #     # TODO bottom left
            #     if dist_bl > 200:
            #         # in lag, find marker in top row that is closest to the current right-most marker
            #         anchor1 = lower_lag[np.argmin(dist.cdist(np.array([bl_dst]), lower_lag, "euclidean")[0])]
            #         # measure how far this marker is away from the right-most marker in the lag image
            #         dist1 = dist.cdist(np.array([anchor1]), np.array([bl_lag]), "euclidean")[0][0]
            #         # get distance between tr and br
            #         dist2 = dist.cdist(np.array([br_lag]), np.array([tr_lag]), "euclidean")[0][0]
            #         result = utils.Intersect2Circles(bl_dst, dist1, tl_dst, dist2)
            #         try:
            #             # identify the point, based on distance to a rough proxy
            #             index = dist.cdist(np.array([[0, save_img.shape[0]]]), np.array(result),
            #                                "euclidean").argmin()
            #             result = result[index]
            #             bl_inferred = np.asarray(result, dtype="int64").tolist()
            #             # tr_inferred = [4400, 0]
            #             dst += [bl_inferred]
            #             src += [bl_ref.tolist()]
            #         except (IndexError, ValueError):
            #             pass

            # # write rotated and translated key points to file for eventual later use
            # dst = np.asarray(dst)
            # data = {"x": dst[:, 0], "y": dst[:, 1]}
            # df = pd.DataFrame(data)
            # df.to_csv(f'{kpts_path}/{image_id}.txt', index=False)

            # write warped key point coordinates to file for eventual later FINAL roi determination
            src = np.asarray(src)
            data = {"x": src[:, 0], "y": src[:, 1]}
            df = pd.DataFrame(data)
            df.to_csv(f'{kpts_path}/{image_id}.txt', index=False)

            # perspective Transform with the first image of the series as destination
            tform_projective = skimage.transform.ProjectiveTransform()  # acceptable, easy to apply
            tform_polynomial = skimage.transform.PolynomialTransform()  # very bad
            tform_piecewise = skimage.transform.PiecewiseAffineTransform()  # very good, but how to handle areas outside of the keypoints?
            try:
                tform_piecewise.estimate(src, dst)
                tform_projective.estimate(src, dst)
                tform_polynomial.estimate(src, dst)
            except ValueError:
                print("could not derive transformation matrix")
                f = open(f"{path_output}/unmatched.txt", 'a')
                f.write(image_id)
                f.close()
                continue

            # Piecewise Affine
            # Save the object to a file
            with open(f'{roi_path}/{image_id}_tform_piecewise.pkl', 'wb') as file:
                pickle.dump(tform_piecewise, file)
            file.close()
            piecewise_warped = skimage.transform.warp(save_img, tform_piecewise, output_shape=(init_roi_height, roi_widths[0]))
            piecewise_warped = skimage.util.img_as_ubyte(piecewise_warped)
            cv2.imwrite(f'{result_pw}/{image_id}.JPG', cv2.cvtColor(piecewise_warped, cv2.COLOR_BGR2RGB))
            # plt.imshow(piecewise_warped)

            # for p in src:
            #     cv2.circle(piecewise_warped, (p[0], p[1]), radius=15, color=(0, 0, 255), thickness=2)

            # # Load the object from a file
            # with open(f'{roi_path}/{image_id}_tform_piecewise.pkl', 'rb') as file:
            #     tform_piecewise = pickle.load(file)
            # transformed_points = tform_piecewise(np.array(src)).astype("uint64")
            # transformed_points = tform_piecewise.inverse(np.array(dst)).astype("uint64")

            # Projective
            roi_loc['transformation_matrix'] = tform_projective.params.tolist()
            projective_warped = skimage.transform.warp(save_img, tform_projective, output_shape=(init_roi_height, roi_widths[0]))
            projective_warped = skimage.util.img_as_ubyte(projective_warped)
            cv2.imwrite(f'{result_proj}/{image_id}.JPG', cv2.cvtColor(projective_warped, cv2.COLOR_BGR2RGB))

            # polynomial
            roi_loc['transformation_matrix'] = tform_polynomial.params.tolist()
            polynomial_warped = skimage.transform.warp(save_img, tform_polynomial, output_shape=(init_roi_height, roi_widths[0]))
            polynomial_warped = skimage.util.img_as_ubyte(polynomial_warped)
            cv2.imwrite(f'{result_poly}/{image_id}.JPG', cv2.cvtColor(projective_warped, cv2.COLOR_BGR2RGB))

            # fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
            # axs[0].imshow(piecewise_warped)
            # axs[0].set_title('leaf')
            # axs[1].imshow(projective_warped)
            # axs[1].set_title('warped')
            # plt.show(block=True)

            # if os.path.getsize(f'{result_path}/{image_id}.JPG') < 700:
            #     print("could not derive transformation matrix")
            #     f = open(f"{path_output}/unmatched.txt", 'a')
            #     f.write(image_id)
            #     f.close()

            del size_outliers

            # # add transformation matrix to the roi localization info
            # roi_loc['transformation_matrix'] = tform.params.tolist()

        with open(f'{roi_path}/{image_id}.json', 'w') as outfile:
            json.dump(roi_loc, outfile)


# reconstruct single images for evaluation

images = glob.glob(f'{result_pw}/*.JPG')