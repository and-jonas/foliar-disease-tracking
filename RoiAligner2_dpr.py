
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
import utils_dpr
import glob
import os
from pathlib import Path
import copy
from scipy.ndimage import distance_transform_edt

import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Qt5Agg')

import skimage
from skimage import transform

import cv2
from multiprocessing import Manager, Process


class RoiAligner:

    def __init__(self, path_labels, path_images, path_output, path_model, n_cpus):
        self.path_labels = Path(path_labels)
        self.path_images = Path(path_images)
        self.path_output = Path(path_output)
        self.path_model = Path(path_model)
        with open(self.path_model, 'rb') as model:
            self.model = pickle.load(model)
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
        leaf_mask_path = sample_output_path / "leaf_mask"
        crop_path = sample_output_path / "crop"
        if create_dirs:
            for p in (kpts_path, overlay_path, roi_path, result_pw, result_proj, preview_path, leaf_mask_path, crop_path):
                p.mkdir(parents=True, exist_ok=True)
        return kpts_path, overlay_path, roi_path, result_pw, result_proj, preview_path, leaf_mask_path, crop_path

    def log_fail(self, image_id, type, reason):
        """
        Writes the image id of failure casees to a txt file
        :param image_id: if of the current image
        :param type: the type of transformation attempted ("piecewise" of "projective")
        """
        f = open(f"{self.path_output}/unmatched_{type}.txt", 'a')
        f.writelines(image_id + " " + str(reason) + "\n")
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

        return label_series[9:10], image_series[9:10]

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

                try:

                    # if j == 8:
                    #     print("stop")

                    # prepare sample work space
                    image_id = os.path.basename(l_series[j]).replace(".txt", "")

                    out_paths = self.get_output_paths(label_series=l_series[j], create_dirs=True)
                    kpts_path, overlay_path, roi_path, result_pw, result_proj, preview_path, leaf_mask_path, crop_path = out_paths

                    print(image_id)

                    # get key point coordinates from YOLO output
                    coords = pd.read_table(l_series[j], header=None, sep=" ")
                    x = coords.iloc[:, 5] * 8192
                    y = coords.iloc[:, 6] * 5464

                    # get image
                    img = Image.open(i_series[j])
                    img = np.array(img)

                    # remove double detections
                    # TODO can this be done during inference via non maximum suppression
                    point_list, x, y = utils.remove_double_detections(x=x, y=y, tol=50)

                    # remove outliers in the key point detections from YOLO errors
                    outliers_x = utils.reject_outliers(x, tol=None, m=3.)  # larger extension, larger variation
                    outliers_y = utils.reject_outliers(y, tol=None, m=2.5)  # smaller extension, smaller variation
                    outliers = outliers_x + outliers_y
                    point_list = np.delete(point_list, outliers, 0)

                    # if too few points detected, skip
                    if len(point_list) < 7:
                        print("Insufficient marks detected. Skipping. ")
                        continue

                    # get minimum area rectangle around retained key points
                    rect = cv2.minAreaRect(point_list)
                    (center, (w, h), angle) = rect

                    # rotate the image about its center
                    if angle > 45:
                        angle = angle - 90
                    rows, cols = img.shape[0], img.shape[1]
                    m_rot = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                    img_rot = cv2.warpAffine(img, m_rot, (cols, rows))

                    # in the reference image, check again for x outliers and get adjusted new bbox
                    if j == 0:
                        point_list = utils.filter_points_x(point_list=point_list, rotation_matrix=m_rot)
                        rect = cv2.minAreaRect(point_list)
                        (center, (w, h), angle) = rect

                    rect = (center, (w + 224, h + 224), angle)

                    # rotate the bounding box about the image's center
                    box = cv2.boxPoints(rect)
                    pts = np.intp(cv2.transform(np.array([box]), m_rot))[0]
                    pts[pts < 0] = 0

                    # order bbox points clockwise
                    pts = utils.order_points(pts)

                    # record roi localization
                    roi_loc = {'rotation_matrix': m_rot.tolist(), 'bounding_box': pts.tolist()}

                    # make crop to run inference on
                    img_cropped = utils.make_inference_crop(pts, img)
                    cv2.imwrite(f'{crop_path}/{image_id}.JPG', cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))

                    # draw key points and bounding box on overlay image as check
                    overlay = utils.make_bbox_overlay(img, point_list, box)
                    cv2.imwrite(f'{overlay_path}/{image_id}.JPG', cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

                    # crop the roi from the rotated image
                    img_crop = img_rot[pts[0][1]:pts[2][1], pts[0][0]:pts[1][0]]
                    current_roi_width = img_crop.shape[1]

                    # log roi width to detect outliers in series
                    size_outliers = None
                    roi_widths.append(current_roi_width)
                    if j == 0:
                        init_roi_height = img_crop.shape[0]

                    # detect size outliers and remove from the logged bbox width values
                    # if j != 0:
                        # size_outliers = utils.reject_size_outliers(data=roi_widths, max_diff=200)
                        # if size_outliers is not None:
                        #     del roi_widths[-1]

                    # copy image for later use
                    save_img = copy.copy(img_crop)

                    # rotate and translate key point coordinates
                    kpts = np.intp(cv2.transform(np.array([point_list]), m_rot))[0]

                    # get tx and ty values for key point translation
                    tx, ty = (-pts[0][0], -pts[0][1])
                    translation_matrix = np.array([
                        [1, 0, tx],
                        [0, 1, ty]
                    ], dtype=np.float32)

                    # apply  translation to key points
                    kpts = np.intp(cv2.transform(np.array([kpts]), translation_matrix))[0]

                    # remove outliers from rotated coordinates,
                    # separately for top, bottom and edge marks
                    reference = True if j == 0 else False
                    w_ref = None if j == 0 else w_ref
                    l, r, t, b, w = utils.identify_outliers_2d(
                        pts=kpts,
                        tol=100,
                        m=3,
                        w_ref=w_ref,
                        reference=reference
                    )
                    w_ref = w if j == 0 else w_ref
                    kpts = l, r, t, b

                    # get leaf mask
                    mask, overlay = utils.segment_image(
                        scale_factor=0.2,
                        img=img_crop,
                        model=self.model
                    )
                    cv2.imwrite(f'{leaf_mask_path}/{image_id}.JPG', cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

                    # export a preview
                    preview = utils.make_bbox_overlay(img=img_crop, pts=(l, r, t, b), box=None)
                    cv2.imwrite(f'{preview_path}/{image_id}.JPG', cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))

                    # get relative x-positions of the edge marks
                    dist = utils.get_leaf_edge_distances(pts=(l, r), leaf_mask=mask)

                    # match all images in the series to the first image where possible
                    # For the first image of each series, no further processing is needed
                    if j == 0:
                        kpts_ref = kpts
                        dist_ref = dist
                        w_ref = w
                        cv2.imwrite(f'{result_pw}/{image_id}.JPG', cv2.cvtColor(save_img, cv2.COLOR_BGR2RGB))
                        cv2.imwrite(f'{result_proj}/{image_id}.JPG', cv2.cvtColor(save_img, cv2.COLOR_BGR2RGB))
                        roi_widths[0] = int(w)

                    # For all subsequent images, perform image registration with the first image as target
                    elif j > 0:
                        # match key points with those on the first image of the series
                        # by searching for the closest points in the top and bottom rows
                        # if non is found in proximity, eliminate from both images
                        src1, dst1 = utils.find_keypoint_matches(
                            current=kpts,
                            current_orig=kpts,
                            ref=kpts_ref,
                            dist_limit=250,
                        )
                        # match key points with those on the first image of the series
                        # by searching for the closest points in the left and right marks
                        # if non is found in proximity, eliminate from both images
                        src2, dst2 = utils.find_distance_matches(
                            current=dist,
                            ref=dist_ref,
                            c_kpt=kpts,
                            r_kpt=kpts_ref,
                            rel_limit=0.1,
                        )
                        # verify that matches are spatially reasonable; remove outlier associations
                        # if too few points detected, skip
                        src1, dst1 = utils.check_keypoint_matches(src=src1, dst=dst1, mdev=50, tol=100, m=2.5)

                        # combine all (l, r, t, b)
                        src = src2 + src1
                        dst = dst2 + dst1

                        # if there are few matches, or if there is a different roi size from the expected,
                        # there is likely a translation due to key point detection errors
                        # try to match with the previous image in the series using SIFT + RANSAC
                        n_kpts = np.sum([len(kpts[i]) for i in range(len(kpts))])
                        n_matches = np.sum([len(src[i]) for i in range(len(src))])
                        match_thresh = int(0.95*n_kpts)
                        if n_matches < match_thresh or np.abs(w-w_ref) > 200:
                            if n_matches < match_thresh:
                                print(" ----- Key point mis-match. Matching on last image in series.")
                            if np.abs(w-w_ref) > 200:
                                print(" ----- Size outlier detected. Matching on last image in series.")

                            # go back (max 4 time steps) in the series until success
                            aligned = False
                            i = 1
                            while not aligned and i < 4:
                                try:
                                    # load previous image
                                    prev_image_id = os.path.basename(l_series[j - i]).replace(".txt", "")
                                    previous_image = Image.open(f'{result_proj}/{prev_image_id}.JPG')
                                except FileNotFoundError:
                                    i += 1
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
                                    res, vis, H = stitcher.stitch(
                                        images=[copy.copy(previous_image), copy.copy(current_image)],
                                        masks=[None, None],
                                        showMatches=True,
                                    )

                                except TypeError:
                                    self.log_fail(image_id, type="projective", reason="no_H")
                                    i += 1
                                    continue

                                # warp key points by applying the inverse of the homography matrix
                                try:
                                    kpts_warped = []
                                    for type in kpts:
                                        kpts_warped.append(np.array([utils.warp_point(x[0], x[1], np.linalg.inv(H)) for x in type]))
                                except:
                                    self.log_fail(image_id, type="projective", reason="no_warped")
                                    i += 1
                                    continue

                                # match key points with those on the first image of the series
                                # by searching for the closest points in the top and bottom rows
                                # if non is found in proximity, eliminate from both images
                                src1, dst1 = utils.find_keypoint_matches(
                                    current=kpts_warped,
                                    current_orig=kpts,
                                    ref=kpts_ref,
                                    dist_limit=250
                                )

                                # match key points with those on the first image of the series
                                # by searching for the closest points in the left and right marks
                                # if non is found in proximity, eliminate from both images
                                src2, dst2 = utils.find_distance_matches(
                                    current=dist,
                                    ref=dist_ref,
                                    c_kpt=kpts,
                                    r_kpt=kpts_ref,
                                    rel_limit=0.1
                                )
                                # verify that matches are spatially reasonable; remove outlier associations
                                # if too few points detected, skip
                                src1, dst1 = utils.check_keypoint_matches(src=src1, dst=dst1, mdev=50, tol=100, m=2.5)

                                # combine all (l, r, t, b)
                                src = src2 + src1
                                dst = dst2 + dst1

                                i += 1

                                # check if now aligned and exit loop
                                if np.sum([len(src[i]) for i in range(len(src))]) > 7:
                                    aligned = True

                        # write warped key point coordinates to file for eventual later FINAL roi determination
                        src_filtered = [np.array(lst) for lst in src if lst]
                        dst_filtered = [np.array(lst) for lst in dst if lst]
                        src_ = np.vstack(src_filtered)
                        dst_ = np.vstack(dst_filtered)
                        try:
                            data = {"x": src_[:, 0], "y": src_[:, 1]}
                            df = pd.DataFrame(data)
                            df.to_csv(f'{kpts_path}/{image_id}.txt', index=False)
                        except IndexError:
                            pass

                        # Transform with the first image of the series as destination
                        # PROJECTIVE
                        tform_projective = transform.ProjectiveTransform()
                        try:
                            tform_projective.estimate(src_, dst_)
                        except:
                            self.log_fail(image_id, type="projective", reason="transform_fail")
                            continue
                        # PIECEWISE
                        tform_piecewise = transform.PiecewiseAffineTransform()
                        try:
                            tform_piecewise.estimate(src_, dst_)
                        except:
                            self.log_fail(image_id, type="piecewise", reason="transform_fail")
                            continue

                        # Save
                        # PROJECTIVE
                        roi_loc['transformation_matrix'] = tform_projective.params.tolist()
                        projective_warped = skimage.transform.warp(save_img, tform_projective,
                                                                   output_shape=(init_roi_height, roi_widths[0]))
                        projective_warped = skimage.util.img_as_ubyte(projective_warped)
                        cv2.imwrite(f'{result_proj}/{image_id}.JPG', cv2.cvtColor(projective_warped, cv2.COLOR_BGR2RGB))
                        # PIECEWISE
                        with open(f'{roi_path}/{image_id}_tform_piecewise.pkl', 'wb') as file:
                            pickle.dump(tform_piecewise, file)
                        file.close()
                        piecewise_warped = transform.warp(save_img, tform_piecewise,
                                                          output_shape=(init_roi_height, roi_widths[0]))
                        piecewise_warped = skimage.util.img_as_ubyte(piecewise_warped)
                        cv2.imwrite(f'{result_pw}/{image_id}.JPG', cv2.cvtColor(piecewise_warped, cv2.COLOR_BGR2RGB))

                        size_outliers = None

                        # add transformation matrix to the roi localization info
                        roi_loc['transformation_matrix'] = tform_projective.params.tolist()

                    with open(f'{roi_path}/{image_id}.json', 'w') as outfile:
                        json.dump(roi_loc, outfile)

                except:
                    self.log_fail(image_id, type="all", reason="unknown")
                    continue

            # monitor progress
            sample_identifier = "_".join(os.path.basename(l_series[0]).replace(".txt", "").split("_")[-2:])
            result.put(sample_identifier)

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
                print("processing " + str(count) + "/" + str(max_jobs))

            for p in processes:
                p.join()