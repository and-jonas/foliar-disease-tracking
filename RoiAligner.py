
# ======================================================================================================================
# Script to align the ROIs from images in a series
# ======================================================================================================================

# import libraries
from ImageStitcher import Stitcher
import numpy as np
import json
import pandas as pd
import pickle
from PIL import Image
import utils
import glob
import os
from pathlib import Path
import copy

# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('Qt5Agg')

import skimage
from skimage import transform

import cv2
from multiprocessing import Manager, Process


class RoiAligner:

    def __init__(self, path_labels, path_images, path_output, n_cpus):
        self.path_labels = Path(path_labels)
        self.path_images = Path(path_images)
        self.path_output = Path(path_output)
        self.n_cpus = n_cpus

    def prepare_workspace(self):
        """
        Creates all required output directories
        """
        self.path_output.mkdir(parents=True, exist_ok=True)
        with open(f"{self.path_output}/unmatched_projective.txt", 'w') as file:
            pass
        with open(f"{self.path_output}/unmatched_piecewise.txt", 'w') as file:
            pass

    def get_output_paths(self, label_series, create_dirs):
        """
        Creates all required output paths and makes the corresponding directories
        :param label_series: The series to process
        :param create_dirs: Wether or not to create directories
        :return: All required paths
        """
        sample_id = "_".join(os.path.basename(label_series).split("_")[2:4]).replace(".txt", "")
        # generate output paths for each sample and create directories
        sample_output_path = self.path_output / sample_id
        kpts_path = sample_output_path / "keypoints"
        overlay_path = sample_output_path / "overlay"
        roi_path = sample_output_path / "roi"
        result_path = sample_output_path / "result"
        result_pw = result_path / "piecewise"
        result_proj = result_path / "projective"
        preview_path = sample_output_path / "preview"
        crop_path = sample_output_path / "crop"
        if create_dirs:
            for p in (kpts_path, overlay_path, roi_path, result_pw, result_proj, preview_path, crop_path):
                p.mkdir(parents=True, exist_ok=True)
        return kpts_path, overlay_path, roi_path, result_pw, result_proj, preview_path, crop_path

    def log_fail(self, image_id, type):
        """
        Writes the image id of failure casees to a txt file
        :param image_id: if of the current image
        :param type: the type of transformation attempted ("piecewise" of "projective")
        """
        f = open(f"{self.path_output}/unmatched_{type}.txt", 'a')
        f.writelines(image_id + "\n")
        f.close()

    def get_series(self):
        """
        Creates two lists of file paths: to key point coordinate files and to images
        for each of the samples monitored over time, stored in date-wise folders.
        :return:
        """
        label_series = []
        image_series = []

        labels = glob.glob(f'{self.path_labels}/*.txt')
        images = glob.glob(f'{self.path_images}/*.JPG')
        label_image_id = ["_".join(os.path.basename(l).split("_")[2:4]).replace(".txt", "") for l in labels]
        image_image_id = ["_".join(os.path.basename(l).split("_")[2:4]).replace(".JPG", "") for l in images]
        uniques = np.unique(label_image_id)

        if len(images) != len(labels):
            raise Exception("list of images and list of coordinate files are not of equal length.")

        print("found " + str(len(uniques)) + " unique sample names")

        for unique_sample in uniques:
            image_idx = [index for index, image_id in enumerate(image_image_id) if unique_sample == image_id]
            label_idx = [index for index, label_id in enumerate(label_image_id) if unique_sample == label_id]
            sample_image_names = [images[i] for i in image_idx]
            sample_labels = [labels[i] for i in label_idx]
            # sort to ensure sequential processing of subsequent images
            sample_image_names = sorted(sample_image_names, key=lambda i: os.path.splitext(os.path.basename(i))[0])
            sample_labels = sorted(sample_labels, key=lambda i: os.path.splitext(os.path.basename(i))[0])
            label_series.append(sample_labels)
            image_series.append(sample_image_names)

        return label_series, image_series

    def process_series(self, work_queue, result):
        """
        Processes the image series for one sample.
        :param work_queue:
        :param result:
        """
        for job in iter(work_queue.get, 'STOP'):

            l_series = job["lseries"]
            i_series = job["iseries"]

            # check that there are an equal number of images and coordinate files
            if len(l_series) != len(i_series):
                print("label series and image series are not of equal length!")
                break

            # iterate over all samples in the series
            roi_widths = []
            for j in range(len(l_series)):

                # prepare sample work space
                image_id = os.path.basename(l_series[j]).replace(".txt", "")
                out_paths = self.get_output_paths(label_series=l_series[j], create_dirs=True)
                kpts_path, overlay_path, roi_path, result_pw, result_proj, preview_path, crop_path = out_paths

                print(image_id)

                # get key point coordinates from YOLO output
                coords = pd.read_table(l_series[j], header=None, sep=" ")
                x = coords.iloc[:, 5] * 8192
                y = coords.iloc[:, 6] * 5464

                # get image
                img = Image.open(i_series[j])
                img = np.array(img)

                # remove double detections
                point_list, x, y = utils.remove_double_detections(x=x, y=y)

                # remove outliers in the key point detections from YOLO errors,
                # get minimum area rectangle around retained key points
                outliers_x = utils.reject_outliers(x, m=3.)  # larger extension, larger variation
                outliers_y = utils.reject_outliers(y, m=2.5)  # smaller extension, smaller variation
                outliers = outliers_x + outliers_y
                point_list = np.delete(point_list, outliers, 0)

                # if too few points detected, skip
                if len(point_list) < 7:
                    print("Insufficient marks detected. Skipping. ")
                    continue

                rect = cv2.minAreaRect(point_list)

                # rotate the image about its center
                (center, (w, h), angle) = rect
                if angle > 45:
                    angle = angle - 90
                rows, cols = img.shape[0], img.shape[1]
                M_img = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                img_rot = cv2.warpAffine(img, M_img, (cols, rows))

                # rotate the bounding box about the image's center
                M_box = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                box = cv2.boxPoints(rect)
                pts = np.intp(cv2.transform(np.array([box]), M_box))[0]
                pts[pts < 0] = 0

                # order bbox points clockwise
                pts = utils.order_points(pts)

                # record roi localization
                roi_loc = {'rotation_matrix': M_img.tolist(), 'bounding_box': pts.tolist()}

                # make crop to run inference on
                img_cropped = utils.make_inference_crop(pts, img)
                cv2.imwrite(f'{crop_path}/{image_id}.JPG', cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))

                # draw key points and bounding box on overlay image as check
                overlay = utils.make_bbox_overlay(img, point_list, box)
                cv2.imwrite(f'{overlay_path}/{image_id}.JPG', cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

                # crop the roi from the rotated image
                img_crop = img_rot[pts[0][1]:pts[2][1], pts[0][0]:pts[1][0]]

                # export a preview
                preview = cv2.resize(img_crop, (0, 0), fx=0.2, fy=0.2)
                cv2.imwrite(f'{preview_path}/{image_id}.JPG', cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))

                # log roi width to detect outliers in series
                roi_widths.append(img_crop.shape[1])
                if j == 0:
                    init_roi_height = img_crop.shape[0]

                # detect size outliers and remove from the logged bbox width values
                if j != 0:
                    size_outliers = utils.reject_size_outliers(roi_widths, max_diff=125)
                    if size_outliers:
                        del roi_widths[-1]

                # copy image for later use
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
                # For the first image of each series, no further processing is needed
                if j == 0:
                    kpts_ref = kpts
                    # kpts_norm_ref = kpts_norm
                    cv2.imwrite(f'{result_pw}/{image_id}.JPG', cv2.cvtColor(save_img, cv2.COLOR_BGR2RGB))
                    cv2.imwrite(f'{result_proj}/{image_id}.JPG', cv2.cvtColor(save_img, cv2.COLOR_BGR2RGB))
                # For all subsequent images, perform image registration with the first image as target
                elif j > 0:
                    # match key points with those on the first image of the series
                    # by searching for the closest points
                    # if non is found in proximity, eliminate from both images
                    src, dst = utils.find_keypoint_matches(kpts, kpts, kpts_ref, dist_limit=150)

                    # verify that matches are spatially reasonable; remove outlier associations
                    # if too few points detected, skip
                    src, dst = utils.check_keypoint_matches(src=src, dst=dst, mdev=50, m=5)

                    # if there are few matches, or if there is a different roi size from the expected,
                    # there is likely a translation due to key point detection errors
                    # try to match with the previous image in the series using SIFT + RANSAC
                    match_thresh = int(0.9*len(kpts))
                    if len(src) < match_thresh or size_outliers:
                        if len(src) < match_thresh:
                            print("Key point mis-match. Matching on last image in series.")
                        if size_outliers:
                            print("Size outlier detected. Matching on last image in series.")
                        try:
                            prev_image_id = os.path.basename(l_series[j - 1]).replace(".txt", "")
                            previous_image = Image.open(f'{result_proj}/{prev_image_id}.JPG')
                        except FileNotFoundError:
                            try:
                                prev_image_id = os.path.basename(l_series[j - 2]).replace(".txt", "")
                                previous_image = Image.open(f'{result_proj}/{prev_image_id}.JPG')
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
                            result, vis, H = stitcher.stitch(
                                images=[copy.copy(previous_image), copy.copy(current_image)],
                                masks=[None, None],
                                showMatches=True,
                            )

                        except TypeError:
                            self.log_fail(image_id, type="projective")
                            continue

                        # warp key points by applying the inverse of the homography matrix
                        kpts_warped = [utils.warp_point(x[0], x[1], np.linalg.inv(H)) for x in kpts]

                        # search for matches with sufficient tolerance
                        src, dst = utils.find_keypoint_matches(kpts_warped, kpts, kpts_ref, dist_limit=200)

                        # verify that matches are spatially reasonable; remove outlier associations
                        src, dst = utils.check_keypoint_matches(src=src, dst=dst, mdev=50, m=4)

                    # write warped key point coordinates to file for eventual later FINAL roi determination
                    try:
                        src = np.asarray(src)
                        data = {"x": src[:, 0], "y": src[:, 1]}
                        df = pd.DataFrame(data)
                        df.to_csv(f'{kpts_path}/{image_id}.txt', index=False)
                    except IndexError:
                        pass

                    # Transform with the first image of the series as destination
                    tform_projective = transform.ProjectiveTransform()
                    tform_piecewise = transform.PiecewiseAffineTransform()
                    try:
                        tform_projective.estimate(src, dst)
                    except:
                        self.log_fail(image_id, type="projective")
                        continue
                    try:
                        tform_piecewise.estimate(src, dst)
                    except:
                        self.log_fail(image_id, type="piecewise")
                        continue

                    # Piecewise Affine
                    # Save the object
                    with open(f'{roi_path}/{image_id}_tform_piecewise.pkl', 'wb') as file:
                        pickle.dump(tform_piecewise, file)
                    file.close()
                    piecewise_warped = transform.warp(save_img, tform_piecewise,
                                                      output_shape=(init_roi_height, roi_widths[0]))
                    piecewise_warped = skimage.util.img_as_ubyte(piecewise_warped)
                    cv2.imwrite(f'{result_pw}/{image_id}.JPG', cv2.cvtColor(piecewise_warped, cv2.COLOR_BGR2RGB))

                    # Projective
                    roi_loc['transformation_matrix'] = tform_projective.params.tolist()
                    projective_warped = skimage.transform.warp(save_img, tform_projective,
                                                               output_shape=(init_roi_height, roi_widths[0]))
                    projective_warped = skimage.util.img_as_ubyte(projective_warped)
                    cv2.imwrite(f'{result_proj}/{image_id}.JPG', cv2.cvtColor(projective_warped, cv2.COLOR_BGR2RGB))

                    del size_outliers

                    # add transformation matrix to the roi localization info
                    roi_loc['transformation_matrix'] = tform_projective.params.tolist()

                with open(f'{roi_path}/{image_id}.json', 'w') as outfile:
                    json.dump(roi_loc, outfile)

    def process_all(self):

        self.prepare_workspace()
        label_series, image_series = self.get_series()

        if len(label_series) > 0:
            # make job and results queue
            m = Manager()
            jobs = m.Queue()
            results = m.Queue()
            processes = []
            # Progress bar counter
            max_jobs = len(label_series)
            count = 0

            # Build up job queue
            for lseries, iseries in zip(label_series, image_series):
                print("to queue")
                job = dict()
                job['lseries'] = lseries
                job['iseries'] = iseries
                jobs.put(job)

            # Start processes
            for w in range(self.n_cpus):
                p = Process(target=self.process_series,
                            args=(jobs, results))
                p.daemon = True
                p.start()
                processes.append(p)
                jobs.put('STOP')

            print(str(len(label_series)) + " jobs started, " + str(self.n_cpus) + " workers")

            # Get results and increment counter along with it
            while count < max_jobs:
                img_names = results.get()
                count += 1
                print("processed " + str(count) + "/" + str(max_jobs))

            for p in processes:
                p.join()

