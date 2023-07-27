
# ======================================================================================================================
# Script to align the ROIs from images in a series
# ======================================================================================================================

# import libraries
from Test import Stitcher
import numpy as np
import json
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
import multiprocessing
from multiprocessing import Manager, Process
import warnings
matplotlib.use('Qt5Agg')


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
            # warnings.warn("list of images and list of coordinate files are not of equal length."
            #               "Ignoring extra coordinate files.")

        print("found " + str(len(uniques)) + " unique sample names")

        for unique_sample in uniques[:48]:
            image_idx = [index for index, image_id in enumerate(image_image_id) if unique_sample == image_id]
            label_idx = [index for index, label_id in enumerate(label_image_id) if unique_sample == label_id]
            sample_image_names = [images[i] for i in image_idx]
            sample_labels = [labels[i] for i in label_idx]
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

                image_id = os.path.basename(l_series[j]).replace(".txt", "")
                sample_id = "_".join(os.path.basename(l_series[j]).split("_")[2:4]).replace(".txt", "")

                print(str(sample_id) + ": image " + str(j))

                # generate output paths for each sample and create directories
                sample_output_path = self.path_output / sample_id
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
                outliers_x = utils.reject_outliers(x)
                outliers_y = utils.reject_outliers(y)
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

                        # try stitching images using SIFT and RANSAC
                        stitcher = Stitcher()
                        try:
                            (result, vis, H) = stitcher.stitch(
                                images=[copy.copy(previous_image), copy.copy(current_image)],
                                masks=[None, None],
                                showMatches=True)
                        except TypeError:
                            print("could not match images")
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
                    tform.estimate(src, dst)
                    warped = skimage.transform.warp(save_img, tform, output_shape=(init_roi_height, roi_widths[0]))
                    warped = skimage.util.img_as_ubyte(warped)
                    cv2.imwrite(f'{result_path}/{image_id}.JPG', cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

                    del size_outliers

                    # add transformation matrix to the roi localization info
                    roi_loc['transformation_matrix'] = tform.params.tolist()

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
