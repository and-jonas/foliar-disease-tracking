
# ======================================================================================================================
# Script to track developing lesions in time series of images and extract data for each lesion
# ======================================================================================================================

from pathlib import Path
import glob
import os.path
import utils
import lesion_utils
import imageio
import numpy as np
import pandas as pd
import cv2
import copy
from scipy import ndimage
from multiprocessing import Manager, Process


class TSAnalyzer:

    def __init__(self, path_aligned_masks,  path_images, path_output, n_cpus):
        self.path_aligned_masks = Path(path_aligned_masks)
        self.path_images = Path(path_images)
        self.path_output = Path(path_output)
        self.n_cpus = n_cpus

    def prepare_workspace(self):
        """
        Creates all required output directories
        """
        self.path_output.mkdir(parents=True, exist_ok=True)

    def create_output_dirs(self, series_id):
        """
        Creates all required output directories for a specific series of images
        :param series_id:
        """
        sample_output_path = self.path_output / series_id
        m_checker_path = sample_output_path / "mask"
        o_checker_path = sample_output_path / "overlay"
        l_data_path = sample_output_path / "lesion_data"
        for p in (m_checker_path, o_checker_path, l_data_path):
            p.mkdir(parents=True, exist_ok=True)
        return m_checker_path, o_checker_path, l_data_path

    def get_series(self):
        """
        Creates two lists of file paths: to key point coordinate files and to images
        for each of the samples monitored over time, stored in date-wise folders.
        :return:
        """
        mask_series = []
        image_series = []

        masks = glob.glob(f'{self.path_aligned_masks}/*.png')
        images = glob.glob(f'{self.path_images}/*.JPG')
        mask_image_id = ["_".join(os.path.basename(l).split("_")[2:4]).replace(".png", "") for l in masks]
        image_image_id = ["_".join(os.path.basename(l).split("_")[2:4]).replace(".JPG", "") for l in images]
        uniques = np.unique(mask_image_id)

        # if len(images) != len(masks):
        #     raise Exception("list of images and list of coordinate files are not of equal length.")
        #     # warnings.warn("list of images and list of coordinate files are not of equal length."
        #     #               "Ignoring extra coordinate files.")

        print("found " + str(len(uniques)) + " unique sample names")

        for unique_sample in uniques:
            image_idx = [index for index, image_id in enumerate(image_image_id) if unique_sample == image_id]
            mask_idx = [index for index, mask_id in enumerate(mask_image_id) if unique_sample == mask_id]
            sample_image_names = [images[i] for i in image_idx]
            sample_masks = [masks[i] for i in mask_idx]
            # sort to ensure sequential processing of subsequent images
            sample_image_names = sorted(sample_image_names, key=lambda i: os.path.splitext(os.path.basename(i))[0])
            sample_masks = sorted(sample_masks, key=lambda i: os.path.splitext(os.path.basename(i))[0])
            mask_series.append(sample_masks)
            image_series.append(sample_image_names)

        return mask_series, image_series

    def process_series(self, work_queue, result):
        """
        Processes the image series for one sample.
        :param work_queue:
        :param result:
        """
        for job in iter(work_queue.get, 'STOP'):

            m_series = job["mseries"]
            i_series = job["iseries"]

            sample_id = os.path.basename(m_series[0])
            sample_id = sample_id.replace(".png", "")
            print("processing " + sample_id)

            # # check that there are an equal number of images and coordinate files
            # if len(m_series) != len(i_series):
            #     print("mask series and image series are not of equal length!")
            #     break

            # generate output directories for each series
            series_id = "_".join(os.path.basename(m_series[0]).split("_")[2:4]).replace(".png", "")
            out_paths = self.create_output_dirs(series_id=series_id)

            # Initialize unique object labels
            next_label = 1
            labels = {}
            all_objects = {}
            num_frames = len(m_series)

            # Process each frame in the time series
            for frame_number in range(1, num_frames + 1):

                # print("processing frame " + os.path.basename(m_series[frame_number - 1]))

                png_name = os.path.basename(m_series[frame_number - 1])
                data_name = png_name.replace(".png", ".txt")

                # ==================================================================================================================
                # 1. Pre-processing
                # ==================================================================================================================

                # Load the multi-class segmentation mask
                frame = cv2.imread(m_series[frame_number - 1], cv2.IMREAD_GRAYSCALE)

                # get leaf mask
                mask_leaf = np.where(frame >= 127, 1, 0).astype("uint8")
                mask_leaf = ndimage.binary_fill_holes(mask_leaf)
                mask_leaf = np.where(mask_leaf, 255, 0).astype("uint8")

                # get lesion mask
                frame = np.where(frame == 191, 255, 0).astype("uint8")
                frame = utils.filter_objects_size(mask=frame, size_th=250, dir="smaller")
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
                img = cv2.imread(i_series[frame_number - 1], cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # init dict
                object_matches = {}  # matches
                objects = {}  # all objects

                # check if there are any missing objects from the last frame in the current frame
                # update the mask if needed
                for lab, (lag_x, lag_y, lag_w, lag_h) in all_objects.items():
                    out1 = seg_lag[lag_y:lag_y + lag_h, lag_x:lag_x + lag_w]
                    out2 = seg[lag_y:lag_y + lag_h, lag_x:lag_x + lag_w]
                    overlap = np.sum(np.bitwise_and(out1, out2)) / (255 * len(np.where(out1)[1]))
                    # if the object cannot be retrieved in the current mask,
                    # paste the object from the previous frame into the current one
                    if overlap < 0.1:
                        seg[lag_y:lag_y + lag_h, lag_x:lag_x + lag_w] = seg_lag[lag_y:lag_y + lag_h,
                                                                        lag_x:lag_x + lag_w]
                # cv2.imwrite(f'{path}/mask_pp/2/{png_name}', seg)

                # ==================================================================================================================
                # 4. Analyze each lesion: label and extract data
                # ==================================================================================================================

                # find contours
                contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                # if not lesions are found, the original image without overlay is saved
                if len(contours) < 1:
                    imageio.imwrite(f'{out_paths[1]}/{png_name}', img)
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

                    # get a mask of the current object in context
                    roi = lesion_utils.select_roi_2(rect=rect, mask=seg)

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

                        # get the ask of the lag object in context
                        roi_lag = lesion_utils.select_roi_2(rect=rect_lag, mask=seg_lag)

                        # get areas and overlap
                        current_area = np.sum(np.logical_and(roi, roi))
                        lag_area = np.sum(np.logical_and(roi_lag, roi_lag))
                        int_area = np.sum(np.logical_and(roi, roi_lag))
                        contour_overlap = int_area / lag_area

                        # get centroid of previous object
                        ctr_lag = np.array([lag_x + lag_w / 2, lag_y + lag_h / 2])

                        # calculate the distance between centroids
                        ctr_dist = np.linalg.norm(current_centroid - ctr_lag)

                        if contour_overlap >= 0.3:  # <==CRITICAL=======================================================
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
                    analyzable_perimeter = len(spl[1]) / len(spl[0])
                    edge_perimeter = len(spl[3]) / len(spl[0])
                    neigh_perimeter = len(spl[2]) / len(spl[0])

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
                result.to_csv(f'{out_paths[2]}/{data_name}', index=False)

                # Draw and save the labeled objects on the frame
                frame_with_labels = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)
                image_with_labels = copy.copy(out_checker)
                for label, (x, y, w, h) in labels.items():
                    cv2.rectangle(frame_with_labels, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame_with_labels, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                                2)
                    cv2.rectangle(image_with_labels, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(image_with_labels, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255),
                                2)
                cv2.imwrite(f'{out_paths[0]}/{png_name}', frame_with_labels)
                imageio.imwrite(f'{out_paths[1]}/{png_name}', image_with_labels)

            # ==================================================================================================================

    def process_all(self):

        self.prepare_workspace()
        mask_series, image_series = self.get_series()

        if len(mask_series) > 0:
            # make job and results queue
            m = Manager()
            jobs = m.Queue()
            results = m.Queue()
            processes = []
            # Progress bar counter
            max_jobs = len(mask_series)
            count = 0

            # Build up job queue
            for mseries, iseries in zip(mask_series, image_series):
                print("to queue")
                job = dict()
                job['mseries'] = mseries
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

            print(str(len(mask_series)) + " jobs started, " + str(self.n_cpus) + " workers")

            # Get results and increment counter along with it
            while count < max_jobs:
                img_names = results.get()
                count += 1
                print("processed " + str(count) + "/" + str(max_jobs))

            for p in processes:
                p.join()
