# ======================================================================================================================
# Various helper functions
# ======================================================================================================================
from builtins import bin

import numpy as np
import cv2
from scipy import ndimage
from scipy.spatial import KDTree
from scipy.spatial import distance as dist
from skimage.feature import peak_local_max
from sklearn.neighbors import KernelDensity
import skimage
import copy
import utils_smoothing
import pickle
from PIL import Image


import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Qt5Agg')


def make_point_list_(input):
    """
    Transform cv2 format to ordinary list of 2D points
    :param input: the point list in cv2 format or as a 2d array
    :return: list of point coordinates
    """
    xs = []
    ys = []
    for point in range(len(input)):
        x = input[point][0]
        y = input[point][1]
        xs.append(x)
        ys.append(y)
    point_list = []
    for a, b in zip(xs, ys):
        point_list.append([a, b])
    c = point_list

    return c


def reject_outliers(data, tol=None, m=2.):
    """
    Detects outliers in 1d and returns the list index of the outliers
    :param data: 1d array
    :param tol: a tolerance in absolute distance
    :param m: number of sd s to tolerate
    :return: list index of outliers
    """
    d = np.abs(data - np.mean(data))
    mdev = np.mean(d)
    s = d / mdev if mdev else np.zeros(len(d))
    idx = np.where(s > m)[0].tolist()  # no difference available for first point - no changes
    if tol is not None:
        abs_diff = np.abs(np.diff(data))
        abs_diff = np.append(abs_diff[0], abs_diff)
        idx = [i for i in idx if abs_diff[i] > tol]  # remove outliers within the absolute tolerance

    return idx


def separate_marks(pts, w_ref, reference):
    """
    Separates top-row and bottom-row marks and sorts from left to right
    :param pts: key points
    :return: separated and sorted mark coordinates
    """

    # separate marks on the left and right edges (where multiple marks may be present)
    # from "inner" marks (where only two separate rows of marks along the leaf edges are expected)
    pts_sorted_x = pts[np.argsort(pts[:, 0]), :]

    # try to find edge markers by looking for at least 3 markers with similar x coordinates on both x ends
    # If not at least 3 are found, delete all candidates
    # except in the reference image, where the left-most points are considered to be the l.
    ref_l = pts_sorted_x[:8, :]
    ref_r = pts_sorted_x[-8:, :]
    left_diff = np.diff(ref_l[:, 0])
    idx1l = np.min(np.where(left_diff < 100)[0])
    if idx1l > 0:
        idx2l = np.where(left_diff > 100)[0][idx1l]
    else:
        idx2l = np.min(np.where(left_diff > 100)[0])
    l_idx = list(range(idx1l, idx2l + 1))
    if not len(l_idx) > 2 and not reference:
        l_idx = []
    right_diff = np.diff(ref_r[:, 0])
    bb = np.where(right_diff < 100)[0]
    xx = np.diff(bb)
    if np.all(xx > 1):
        idx1r = len(pts_sorted_x)
    elif any(xx > 1):
        pos = len(np.where(xx > 1)[0])
        idx1r = np.min(bb[np.where(xx > 1)[0][0] + pos:]) + len(pts_sorted_x) - 8
    else:
        idx1r = np.min(bb) + len(pts_sorted_x) - 8
    idx2r = np.max(np.where(right_diff < 100)[0]) + 2 + len(pts_sorted_x) - 8
    r_idx = list(range(idx1r, idx2r))
    # adjust for potentially removed marks
    if len(l_idx) > 0:
        r_idx = [i-idx1l for i in r_idx]
    if not len(r_idx) > 2:
        r_idx = []

    # if edge markers are found, delete any markers that lie outside the putative edge markers
    if len(l_idx) > 0:
        pts_sorted_x = pts_sorted_x[idx1l:]
        l_idx = [i - idx1l for i in l_idx]
    if len(r_idx) > 0:
        pts_sorted_x = pts_sorted_x[:idx2r]

    # if no edge markers are found (maybe only 1-2 of them left)
    # --> if not a size outlier, try to match by x - position
    # get minimum area rectangle around retained key points
    # get current roi width
    rect = cv2.minAreaRect(pts_sorted_x)
    (center, (w, h), angle) = rect
    w = max(w, h)  # w and h can be exchanged!?!?!
    w += 224
    if not reference:
        # check if roi width matches
        if np.abs(w - w_ref) < 200:
            left_position = 112
            right_position = w_ref - 112
            if len(l_idx) == 0:
                l_idx = np.where(np.abs(pts_sorted_x[:, 0] - left_position) < 200)[0]
            if len(r_idx) == 0:
                r_idx = np.where(np.abs(pts_sorted_x[:, 0] - right_position) < 200)[0]

    # get index of the inner (t, b) marks
    if len(l_idx) == 0:
        in_start = 0
    else:
        in_start = l_idx[-1] + 1
    if len(r_idx) == 0:
        in_end = len(pts)
    else:
        in_end = r_idx[0]
    in_idx = np.array(range(in_start, in_end))

    # select the points
    pts_left = pts_sorted_x[l_idx]
    l = pts_left[np.argsort(pts_left[:, 1])]
    pts_right = pts_sorted_x[r_idx]
    r = pts_right[np.argsort(pts_right[:, 1])]
    pts_inner = pts_sorted_x[in_idx]
    pts_inner = pts_inner[np.argsort(pts_inner[:, 1])]

    # Use polynomial to separate top and bottom marks
    coefficients = np.polyfit(pts_inner[:, 0], pts_inner[:, 1], deg=2)
    y_predicted = np.polyval(coefficients, pts_inner[:, 0])
    residuals = pts_inner[:, 1] - y_predicted

    # Find top and bottom points, using the residuals
    # sort from left to right
    t_idx = np.where(residuals < 0)[0]
    b_idx = np.where(residuals > 0)[0]
    t = pts_inner[t_idx]
    t = t[np.argsort(t[:, 0]), :]
    b = pts_inner[b_idx]
    b = b[np.argsort(b[:, 0]), :]

    return l, r, t, b, w


def identify_outliers_2d(pts, tol, m, w_ref, reference):
    """
    Separates top and bottom points and performs filtering within each group based on y-coordinates
    :param pts: the set of points to split and clean from outliers
    :param tol: the maximum distance to be tolerated
    :param m: the number of sds to tolerate
    :return: the separated top and bottom points, cleaned from outliers
    """

    l, r, t, b, w = separate_marks(pts, w_ref, reference)

    # find top and bottom outliers
    bottom_outliers = reject_outliers(data=b[:, 1], tol=tol, m=m)
    top_outliers = reject_outliers(data=t[:, 1], tol=tol, m=m)

    # clean by removing detected outliers
    t = np.delete(t, top_outliers, 0)
    b = np.delete(b, bottom_outliers, 0)

    return l, r, t, b, w


def pairwise_distances(points1, points2):
    """
    Calculates the distances between pairs of associated points in x-axis (y-axis is ignored)
    :param points1: 2d coordinates of points
    :param points2: 2d coordinates of points
    :return: distances for each pair of associated points
    """
    distances = []

    for p1, p2 in zip(points1, points2):
        distance = p1[0] - p2[0]
        distances.append(distance)

    return distances


def reject_size_outliers(data, max_diff):
    """
    Detects outliers in 1d and returns the list index of the outliers
    :param max_diff: size difference threshold in px
    :param data: 1d array
    :return: list index of outliers
    """
    mean_size_prev = np.mean(data[:-1])
    current_size = data[-1]
    if np.abs(current_size - mean_size_prev) > max_diff:
        idx = len(data) - 1
        if current_size - mean_size_prev > 0:
            direction = "greater"
        else:
            direction = "smaller"

        return [idx, direction]
    else:
        return None


def filter_points(x, y, min_distance):
    """
    Removes all but one point if multiple are close-by
    :param x: x-coordinates of points
    :param y: y-coordinates of points
    :param min_distance: minimum distance between points required for them to be both maintained
    :return: filtered points
    """
    points = np.array([[a, b] for a, b in zip(x, y)], dtype=np.int32)

    filtered_points = []
    remaining_points = points.copy()

    while len(remaining_points) > 0:
        current_point = remaining_points[0]
        remaining_points = np.delete(remaining_points, 0, axis=0)
        filtered_points.append(current_point)
        distances = np.linalg.norm(remaining_points - current_point, axis=1)
        remaining_points = remaining_points[distances >= min_distance]

    return np.array(filtered_points)


def remove_double_detections(x, y, tol):
    """
    Removes one of two coordinate pairs if their distance is below 50
    :param x: x-coordinates of points
    :param y: y-coordinates of points
    :param tol: minimum distance required for both points to be retained
    :return: the filtered list of points and their x and y coordinates
    """
    point_list = np.array([[a, b] for a, b in zip(x, y)], dtype=np.int32)
    dist_mat = dist.cdist(point_list, point_list, "euclidean")
    np.fill_diagonal(dist_mat, np.nan)
    dbl_idx = np.where(dist_mat < tol)[0].tolist()[::2]
    point_list = np.delete(point_list, dbl_idx, axis=0)
    x = np.delete(x, dbl_idx, axis=0)
    y = np.delete(y, dbl_idx, axis=0)
    return point_list, x, y


def make_bbox_overlay(img, pts, box):
    """
    Creates an overlay on the original image that shows the detected marks and the fitted bounding box
    :param img: original image
    :param pts: list of coordinate [x,y] pairs denoting the detected mark positions
    :param box: the box coordinates in cv2 format
    :return: image with overlay
    """
    overlay = copy.copy(img)
    if type(pts) is tuple:
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0)]
        for i in range(len(pts)):
            for point in pts[i]:
                cv2.circle(overlay, (point[0], point[1]), radius=15, color=colors[i], thickness=9)
    else:
        for point in pts:
            cv2.circle(overlay, (point[0], point[1]), radius=15, color=(0, 0, 255), thickness=9)
    if box is not None:
        box_ = np.intp(box)
        cv2.drawContours(overlay, [box_], 0, (255, 0, 0), 9)
    overlay = cv2.resize(overlay, (0, 0), fx=0.25, fy=0.25)
    return overlay


def make_inference_crop(pts, img):
    """
    Makes a crop of the full image that contains the leaf to speed up inference
    :param pts: coordinates of the bounding box
    :param img: image to crop
    :return: cropped image
    """
    # get the centroid
    mw, mh = np.mean(pts, axis=0)

    # crop according to centroid
    h_min = int(mh) - int(2048 / 2)

    if h_min < 0:
        print("Bounding Box outside of the image")

    if h_min + 2048 >= len(img):
        print("Bounding Box outside of the image")

    img_cropped = img[h_min:h_min + 2048, :, :]

    return img_cropped


def warp_point(x: int, y: int, M) -> [int, int]:
    """
    Applies a homography matrix to a point
    :param x: the x coordinate of the point
    :param y: the y coordinates of the point
    :param M: the homography matrix
    :return: coordinates of the warped point
    """
    d = M[2, 0] * x + M[2, 1] * y + M[2, 2]

    return ([
        int((M[0, 0] * x + M[0, 1] * y + M[0, 2]) / d),  # x
        int((M[1, 0] * x + M[1, 1] * y + M[1, 2]) / d),  # y
    ])


def remove_points_from_mask(mask, classes):
    """
    Removes predicted pycnidia and rust pustules from the mask. Replaces the relevant pixel values with the average
    of the surrounding pixels. Points need to be transformed separately and added again to the transformed mask.
    :param mask: the mask to remove the points from
    :param classes: ta list with indices of the classes that are represented as points
    :return: mask with key-points removed
    """

    mask = copy.copy(mask)
    for cl in classes:
        idx = np.where(mask == cl)
        y_points, x_points = idx
        for i in range(len(y_points)):
            row, col = y_points[i], x_points[i]
            surrounding_pixels = mask[max(0, row - 1):min(row + 2, mask.shape[0]),
                                 max(0, col - 1):min(col + 2, mask.shape[1])]
            average_value = np.mean(surrounding_pixels)
            mask[row, col] = average_value
    return mask


def filter_points_x(point_list, image):

    # get minimum area rectangle around retained key points
    rect = cv2.minAreaRect(point_list)

    # enlarge to enable feature extraction for 56 px square box around detected markers
    (center, (w, h), angle) = rect

    # rotate the image about its center
    if angle > 45:
        angle = angle - 90
    rows, cols = image.shape[0], image.shape[1]
    M_img = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

    all_pts_rot = np.intp(cv2.transform(np.array([point_list]), M_img))[0]
    l, r, t, b, w = identify_outliers_2d(
        pts=all_pts_rot,
        tol=100,
        m=3,
        w_ref=None,
        reference=True
    )
    filtered_pts_rot = np.vstack([l, r, t, b])

    # 1. Extract the rotation part (top-left 2x2)
    R = M_img[:2, :2]

    # 2. Transpose the rotation part (this is the inverse rotation)
    R_inv = R.T

    # 3. Compute the inverse translation
    t = M_img[:2, 2]  # Translation vector
    t_inv = -R_inv @ t  # Apply inverse rotation to the negative translation

    # 4. Construct the full inverse affine matrix
    m_rot_inv = np.eye(3)  # Start with identity matrix
    m_rot_inv[:2, :2] = R_inv  # Set the inverse rotation
    m_rot_inv[:2, 2] = t_inv  # Set the inverse translation

    # 5. Apply the inverse transformation to your rotated points (all_pts_rot)
    # Make sure the points are in the right format (array of points, 2D)
    filtered_pts_unrot = cv2.transform(np.array([filtered_pts_rot]), m_rot_inv)[0][:, :2]

    # convert to list and back to np.array
    # this is necessary for cv.minAreaRect() to be able to process the point list (?????)
    filtered_pts_unrot = filtered_pts_unrot.tolist()
    filtered_pts_unrot = np.array(filtered_pts_unrot)

    return filtered_pts_unrot


def rotate_translate_warp_points(mask, classes, rot, box, tf, target_shape, warped):
    """
    rotates, translates, and warps points to match the transformed segmentation mask.
    Filters detected point lying outside the roi.
    :param mask: The original full-sized segmentation mask that includes all classes
    :param classes: List of integers specifying the class of point labels
    :param rot: rotation matrix applied to the msak
    :param box: the corner coordinates of the bounding box used to crop he roi from the image
    :param tf: the transformation matrix
    :param target_shape: the dimension of the desired output image
    :param warped: the warped segmentation mask of the roi, without the points
    :return: The complemented warped roi
    """

    # get input shape
    w = box[1, 0] - box[0, 0]
    h = box[2, 1] - box[1, 1]
    input_shape = (h, w)

    # loop over classes that are represented as points
    for cl in classes:

        # get class pixel positions
        idx = np.where(mask == cl)

        # if there are any pixels to transform, do so, else leave unchanged
        if len(idx[0]) == 0:
            continue

        # extract points
        points = np.array([[a, b] for a, b in zip(idx[1], idx[0])], dtype=np.int32)

        # rotate points
        points_rot = np.intp(cv2.transform(np.array([points]), rot))[0]

        # translate points
        tx, ty = (-box[0][0], -box[0][1])
        translation_matrix = np.array([
            [1, 0, tx],
            [0, 1, ty]
        ], dtype=np.float32)
        points_trans = np.intp(cv2.transform(np.array([points_rot]), translation_matrix))[0]

        # remove any rotated and translated point outside the roi
        mask_pw = (points_trans[:, 1] < input_shape[0]) & (points_trans[:, 1] > 0) & \
                  (points_trans[:, 0] < input_shape[1]) & (points_trans[:, 0] > 0)
        points_filtered = points_trans[mask_pw]

        # create and warp the point mask
        point_mask = np.zeros(input_shape).astype("uint8")
        point_mask[points_filtered[:, 1], points_filtered[:, 0]] = 255
        lm = np.stack([point_mask, point_mask, point_mask], axis=2)
        warped_pycn_mask = skimage.transform.warp(lm, tf, output_shape=target_shape)
        coordinates = peak_local_max(warped_pycn_mask[:, :, 0], min_distance=1)
        warped[coordinates[:, 0], coordinates[:, 1]] = cl

    return warped


def find_keypoint_matches(current, current_orig, ref, dist_limit=150):
    """
    Finds pairs of matching detected marks on two subsequent images of a series
    :param current: the current image to be aligned to the initial image
    :param current_orig: the initial image of the series
    :param ref: the coordinates of keypoints in the reference image
    :param dist_limit: the maximum allowed distance between points to consider them the same point
    :return: matched pairs of keypoints coordinates in the source and the target
    """

    # # separate marks according to position
    # current_sep = separate_marks(pts=current, roi_width=roi_width)
    # current_orig_sep = separate_marks(pts=current_orig, roi_width=roi_width)
    # ref_sep = separate_marks(ref, roi_width=roi_width)

    src = []
    dst = []
    for c, co, r in zip(current[2:], current_orig[2:], ref[2:]):

        # make and query tree
        tree = KDTree(c)
        assoc = []
        for I1, point in enumerate(r):
            _, I2 = tree.query(point, k=1, distance_upper_bound=dist_limit)
            assoc.append((I1, I2))

        # match indices back to key point coordinates
        assocs = []
        for a in assoc:
            p1 = r[a[0]].tolist()
            try:
                p2 = co[a[1]].tolist()
            except IndexError:
                p2 = [np.NAN, np.NAN]
            assocs.append([p1, p2])

        # reshape to list of corresponding source and target key point coordinates
        pair = assocs
        src.append([[*p[0]] for p in pair if p[1][0] is not np.nan])
        dst.append([[*p[1]] for p in pair if p[1][0] is not np.nan])

    return src, dst


def get_leaf_edge_distances(pts, leaf_mask):

    # unpack points
    l, r = pts

    # invert leaf mask
    mask_invert = np.bitwise_not(leaf_mask)

    # get relative positions of left marks
    try:
        if len(l) > 0:
            l_min_x = np.min(np.where(mask_invert[:, np.mean(l[:, 0]).astype(int)] == 255))
            l_max_x = np.max(np.where(mask_invert[:, np.mean(l[:, 0]).astype(int)] == 255))
            l_dist = np.array([(l[i, 1] - l_min_x) / (l_max_x - l_min_x) for i in range(len(l))])
        else:
            l_dist = np.array([])
    except IndexError:
        l_dist = np.array([])

    # get relative positions of right marks
    try:
        if len(r) > 0:
            r_min_x = np.min(np.where(mask_invert[:, np.mean(r[:, 0]).astype(int)] == 255))
            r_max_x = np.max(np.where(mask_invert[:, np.mean(r[:, 0]).astype(int)] == 255))
            r_dist = np.array([(r[i, 1] - r_min_x) / (r_max_x - r_min_x) for i in range(len(r))])
        else:
            r_dist = np.array([])
    except IndexError:
        r_dist = np.array([])

    # assemble output
    dist = (l_dist, r_dist)

    return dist


def find_distance_matches(current, ref, c_kpt, r_kpt, rel_limit):

    # # separate marks according to position
    # c_sep = separate_marks(pts=c_kpt, roi_width=roi_width)
    # r_sep = separate_marks(pts=r_kpt, roi_width=roi_width)

    src = []
    dst = []
    # for left and ride marks
    for i in range(len(current)):
        # subset the relevant marks
        c = current[i]
        r = ref[i]
        assoc = []
        # Compute the pairwise differences
        pairwise_diff = np.abs(c[:, np.newaxis] - r)
        for x, row in enumerate(pairwise_diff):
            min_index = np.argmin(row)
            min_value = row[min_index]
            if min_value < rel_limit:
                assoc.append([min_index, x])
                # assoc.append([x, min_index])

        # match indices back to key point coordinates
        assocs = []
        for a in assoc:
            p1 = r_kpt[i][a[0]].tolist()
            try:
                p2 = c_kpt[i][a[1]].tolist()
            except IndexError:
                p2 = [np.NAN, np.NAN]
            assocs.append([p1, p2])

        # reshape to list of corresponding source and target key point coordinates
        pair = assocs
        src.append([[*p[0]] for p in pair if p[1][0] is not np.nan])
        dst.append([[*p[1]] for p in pair if p[1][0] is not np.nan])

    return src, dst


def check_keypoint_matches(src, dst, mdev, tol, m):
    """
    Verifies that the kd-tree identified associations are meaningful by comparing the distance between source and target
    across the top and bottom rows. Regular patterns are expected, and outliers from this pattern are removed.
    If no stable pattern is found, all associations are deleted.
    :param src: source point coordinates
    :param dst: destination point coordinates
    :param mdev: average deviation from mean that is tolerated for associations
    :param tol: value below which matches are kept even if dev is higher than the specified threshold
    :param m: parameter for outlier removal
    :return: cleaned lists of source and destination points
    """

    # unpack
    src_t, src_b = src
    src_ = src_t + src_b
    dst_t, dst_b = dst
    dst_ = dst_t + dst_b

    if len(src_) < 7:
        src, dst = [], []
    else:
        # broadly check for a regular pattern, if none is found delete all associations
        distances = pairwise_distances(src_, dst_)
        d = np.abs(distances - np.mean(distances))
        m_dev = np.mean(d)
        if mdev is not None and m_dev > mdev:
            src, dst = [], []
        else:
            # otherwise, separately evaluate pairwise distances for top and bottom marks
            # eliminate outliers from both, source and target, if any found
            for type in [src, dst]:
                t, b = type
                t_distances = distances[:len(t)]
                b_distances = distances[len(t):]
                outliers_t = reject_outliers(data=t_distances, tol=tol, m=m)
                outliers_b = reject_outliers(data=b_distances, tol=tol, m=m)
            try:
                src[0] = np.delete(src[0], outliers_t, 0).tolist()
                src[1] = np.delete(src[1], outliers_b, 0).tolist()
                dst[0] = np.delete(dst[0], outliers_t, 0).tolist()
                dst[1] = np.delete(dst[1], outliers_b, 0).tolist()
            except IndexError:
                pass

    return src, dst


def order_points(pts):
    """
    Orders a list of points clock-wise
    :param pts: List of point coordinates pairs as [x, y]
    :return: the coordinates of the top-left, top-right, bottom-right, and bottom-left points
    """
    # sort the points based on their x-coordinates
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points
    left_most = x_sorted[:2, :]
    right_most = x_sorted[-2:, :]
    # sort the left-most coordinates according to their
    # y-coordinates, to grab the top-left and bottom-left points
    leftMost = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = leftMost
    # use tl as anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], right_most, "euclidean")[0]
    (br, tr) = right_most[np.argsort(D)[::-1], :]
    return np.array([tl, tr, br, bl], dtype="int")


def make_point_list(input):
    """
    Transform cv2 format to ordinary point list
    :param input: the list of points to transform
    :return: list of point coordinates
    """
    xs = []
    ys = []
    for point in range(len(input)):
        x = input[point][0]
        y = input[point][1]
        xs.append(x)
        ys.append(y)
    point_list = []
    for a, b in zip(xs, ys):
        point_list.append(tuple([a, b]))
    c = point_list

    return c


def flatten_contour_data(input_contour, asarray, as_point_list=True):
    """
    Extract contour points from cv2 format into point list
    :param input_contour: The cv2 contour to extract
    :param asarray: Boolean, whether output should be returned as an array
    :param as_point_list: Boolean, whetheer output should be returned as a point list
    :return: array or list containing the contour point coordinate pairs
    """
    xs = []
    ys = []
    for point in input_contour[0]:
        x = point[0][1]
        y = point[0][0]
        xs.append(x)
        ys.append(y)
    if as_point_list:
        point_list = []
        # for a, b in zip(xs, ys):
        for a, b in zip(ys, xs):
            point_list.append([a, b])
            c = point_list
        if asarray:
            c = np.asarray(point_list)
        return c
    else:
        return xs, ys


def make_cv2_formatted(array):
    """
    Takes a 2D array of X and Y coordinates and returns a point list in cv2 fomat
    :param array: 2d array with X and Y coordinates
    :return: contour in cv2 format
    """
    # get the points to a list
    L = []
    for p in range(len(array[0])):
        L.append([int(array[1][p]), int(array[0][p])])
    # reshape to cv2 format
    sm_contour = np.array(L).reshape((-1, 1, 2)).astype(np.int32)
    return sm_contour


def filter_objects_size(mask, size_th, dir):
    """
    Filter objects in a binary mask by size
    :param mask: A binary mask to filter
    :param size_th: The size threshold used to filter (objects GREATER than the threshold will be kept)
    :return: A binary mask containing only objects greater than the specified threshold
    """
    _, output, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sizes = stats[1:, -1]
    if dir == "greater":
        idx = (np.where(sizes > size_th)[0] + 1).tolist()
    if dir == "smaller":
        idx = (np.where(sizes < size_th)[0] + 1).tolist()
    out = np.in1d(output, idx).reshape(output.shape)
    cleaned = np.where(out, 0, mask)

    return cleaned


def is_multi_channel_img(img):
    """
    Checks whether the supplied image is multi- or single channel (binary mask or edt).
    :param img: The image, binary mask, or edt to process.
    :return: True if image is multi-channel, False if not.
    """
    if len(img.shape) > 2 and img.shape[2] > 1:
        return True
    else:
        return False


def rectangles_overlap(rect1, rect2):
    """
    Determines if two bboxes overlap
    :param rect1: coordinates of bbox
    :param rect2: coordinates of bbox
    :return: Bool
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Check for no overlap in any direction
    if x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1:
        return False
    else:
        return True


def get_pseudo_leaf_mask(img):
    img_resized = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
    lower = np.array([0, 0, 150])  # v changed successively from 35 to 30 to 22 for selected images
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    M = cv2.medianBlur(mask, 13)
    M_resized = cv2.resize(M, (0, 0), fx=10, fy=10)
    return M_resized


def get_color_spaces(patch):

    # Scale to 0...1
    img_RGB = np.array(patch / 255, dtype=np.float32)

    # Images are in RGBA mode, but alpha seems to be a constant - remove to convert to simple RGB
    img_RGB = img_RGB[:, :, :3]

    # Convert to other color spaces
    img_HSV = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HSV)
    img_Luv = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2Luv)
    img_Lab = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2Lab)
    img_YUV = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YUV)
    img_YCbCr = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YCrCb)

    # Calculate vegetation indices: ExR, ExG, TGI
    R, G, B = cv2.split(img_RGB)
    normalizer = np.array(R + G + B, dtype=np.float32)
    # Avoid division by zero
    normalizer[normalizer == 0] = 10
    r, g, b = (R, G, B) / normalizer

    # weights for TGI
    lambda_r = 670
    lambda_g = 550
    lambda_b = 480

    TGI = -0.5 * ((lambda_r - lambda_b) * (r - g) - (lambda_r - lambda_g) * (r - b))
    ExR = np.array(1.4 * r - b, dtype=np.float32)
    ExG = np.array(2.0 * g - r - b, dtype=np.float32)

    # Concatenate all
    descriptors = np.concatenate(
        [img_RGB, img_HSV, img_Lab, img_Luv, img_YUV, img_YCbCr, np.stack([ExG, ExR, TGI], axis=2)], axis=2)
    # Names
    descriptor_names = ['sR', 'sG', 'sB', 'H', 'S', 'V', 'L', 'a', 'b',
                        'L', 'u', 'v', 'Y', 'U', 'V', 'Y', 'Cb', 'Cr', 'ExG', 'ExR', 'TGI']

    # Return as tuple
    return (img_RGB, img_HSV, img_Lab, img_Luv, img_YUV, img_YCbCr, ExG, ExR, TGI), descriptors, descriptor_names


def segment_image(img, model, scale_factor):
    """
    Segments an image using a pre-trained pixel classification model.
    Creates probability maps, binary segmentation masks, and overlay
    :param img: The image to be processed.
    :return: The resulting binary segmentation mask.
    """

    model.n_jobs = 1

    # resize image
    img_rsz = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)

    # extract pixel features
    color_spaces, descriptors, descriptor_names = get_color_spaces(img_rsz)
    descriptors_flatten = descriptors.reshape(-1, descriptors.shape[-1])

    # extract pixel label probabilities
    segmented_flatten_probs = model.predict_proba(descriptors_flatten)[:, 0]

    # restore image
    probabilities = segmented_flatten_probs.reshape((descriptors.shape[0], descriptors.shape[1]))

    # perform edge-aware smoothing
    output_solver, thresh = utils_smoothing.smooth_edge_aware(reference=img_rsz, target=probabilities)

    ret, thresh1 = cv2.threshold(output_solver, 127, 255, cv2.THRESH_BINARY +
                                 cv2.THRESH_OTSU)

    # get white
    lower = np.array([155, 155, 155])  # v changed successively from 35 to 30 to 22 for selected images
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(img_rsz, lower, upper)

    # blur
    full = np.bitwise_or(thresh1, mask)
    full = cv2.medianBlur(full, 17)
    # fill holes
    bin = ndimage.binary_fill_holes(np.bitwise_not(full))
    full = np.bitwise_not(bin).astype("uint8")
    kernel = np.ones((1, 9), np.uint8)
    full = cv2.dilate(full, kernel, iterations=3)
    full_inv = np.bitwise_not(full)
    full_inv = cv2.dilate(full_inv, kernel, iterations=3)
    full = np.bitwise_not(full_inv)*255

    M = full.ravel()
    M = np.expand_dims(M, -1)
    out_mask = np.dot(M, np.array([[1, 0, 0, 0.33]]))
    out_mask = np.reshape(out_mask, newshape=(img_rsz.shape[0], img_rsz.shape[1], 4))
    out_mask = out_mask.astype("uint8")
    mask = Image.fromarray(out_mask, mode="RGBA")
    img_ = Image.fromarray(img_rsz, mode="RGB")
    img_ = img_.convert("RGBA")
    img_.paste(mask, (0, 0), mask)
    overlay = np.asarray(img_)

    # scale up the mask
    full = cv2.resize(full, (0, 0), fx=1/scale_factor, fy=1/scale_factor, interpolation=cv2.INTER_NEAREST)

    return full, overlay


def process_leaf_mask(img, path_leaf_mask):

    # read mask from leaf-toolkit
    mask = Image.open(path_leaf_mask)
    mask = np.asarray(mask)
    mask = cv2.resize(mask, (0, 0), fx=0.2, fy=0.2)
    img_rsz = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)

    # binarize mask
    mask_bin = np.where(mask != 0, 255, 0)
    mask_bin = mask_bin.astype(np.uint8)

    # # Define a kernel size. Adjust based on the size of deformations.
    # kernel = np.ones((1, 15), np.uint8)
    #
    # # Perform opening (erosion followed by dilation)
    # mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=9)

    # post-process
    mask_pp = cv2.medianBlur(mask_bin, 9)  # blur
    mask_pp = ndimage.binary_fill_holes(mask_pp)  # fill holes
    mask_pp = mask_pp.astype(np.uint8) * 255

    # select largest object
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_pp, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        m = np.zeros_like(mask_pp)
        m[labels == largest_label] = 255
    else:
        m = mask_pp.copy()

    M = m.ravel()
    M = np.expand_dims(M, -1)
    out_mask = np.dot(M, np.array([[1, 0, 0, 0.33]]))
    out_mask = np.reshape(out_mask, newshape=(m.shape[0], m.shape[1], 4))
    out_mask = out_mask.astype("uint8")
    mask = Image.fromarray(out_mask, mode="RGBA")
    img_ = Image.fromarray(img_rsz, mode="RGB")
    img_ = img_.convert("RGBA")
    img_.paste(mask, (0, 0), mask)
    overlay = np.asarray(img_)

    return m, overlay


def split_consecutive_sets(numbers):
    sets = []
    current_set = [numbers[0]]  # Start the first set with the first number

    for i in range(1, len(numbers)):
        if numbers[i] - numbers[i - 1] > 1:  # Check for a gap
            sets.append(current_set)  # Save the current set
            current_set = []  # Start a new set
        current_set.append(numbers[i])

    sets.append(current_set)  # Add the last set
    return sets


def get_pycn_features(mask, lesion_mask, contour, max_dist, bandwidth, kernel):

    # fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    # axs[0].imshow(lesion_binary)
    # axs[0].set_title('mask')
    # axs[1].imshow(mask)
    # axs[1].set_title('density')
    # plt.show(block=True)

    # binarize pycnidia, multiply with lesion mask
    pycnidia_binary = np.uint8(np.where(mask == 212, 1, 0) * lesion_mask / 255)

    # pycnidia coordinates
    coords = np.where(pycnidia_binary == 1)
    coords = np.array(list(zip(coords[0], coords[1])))

    if len(coords) == 0:
        keys = ["frac_pycn", "mean_dist", "variance_dist", "min_dist", "max_dist", "median_dist",
                "mean_l_density", "variance_l_density", "max_l_density", "median_l_density",
                "mean_p_density", "variance_p_density", "max_p_density", "median_p_density"]
        return {key: np.nan for key in keys}, None
    else:

        # (1) DISTANCE
        # get contour distance values
        dmap = ndimage.distance_transform_edt(1 - pycnidia_binary)
        contour_points = contour[:, 0, :]
        dists = []
        for point in contour_points:
            x, y = np.round(point).astype(int)
            if 0 <= x < dmap.shape[1] and 0 <= y < dmap.shape[0]:  # Ensure within bounds
                dists.append(dmap[y, x])
        dists = np.asarray(dists)

        # get distance features
        pycn_contour = np.where(dists <= max_dist)[0]
        density_array = np.array(dists)
        distance_features = {
            "frac_pycn": len(pycn_contour) / len(contour),
            "mean_dist": np.mean(density_array),
            "variance_dist": np.var(density_array),
            "min_dist": np.min(density_array),
            "max_dist": np.max(density_array),
            "median_dist": np.median(density_array)
        }

        # (2) DENSITY
        # get kernel density estimate
        kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
        kde.fit(coords)

        # resize for faster processing
        height = int(lesion_mask.shape[0] / 5)
        width = int(lesion_mask.shape[1] / 5)
        x = np.linspace(0, lesion_mask.shape[1] - 1, width)  # Match resized grid
        y = np.linspace(0, lesion_mask.shape[0] - 1, height)
        x, y = np.meshgrid(x, y)
        grid_coords = np.vstack([y.ravel(), x.ravel()]).T  # Note: (y, x) for consistency

        # Evaluate KDE on the grid
        log_density = kde.score_samples(grid_coords)
        density = np.exp(log_density).reshape(height, width)
        density_rsz = cv2.resize(density, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        # ~ per mm^2 (assuming ~0.03 mm / px)
        density_rsz = density_rsz * 1000

        # mask everything except lesion
        binary_mask = lesion_mask.astype(bool)
        density_array = density_rsz[binary_mask]

        # get density features
        lesion_density_features = {
            "mean_l_density": np.mean(density_array),
            "variance_l_density": np.var(density_array),
            "min_l_density": np.min(density_array),
            "max_l_density": np.max(density_array),
            "median_l_density": np.median(density_array)
        }

        # get a pycnidiation density contour
        pycnidiation_mask = np.where(density_rsz >= 0.0001, 1, 0)
        lesion_pycn_mask = np.logical_and(lesion_mask, pycnidiation_mask)
        pycn_contour, _ = cv2.findContours(np.uint8(lesion_pycn_mask * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        binary_mask = lesion_pycn_mask.astype(bool)
        density_array = density_rsz[binary_mask]

        # get density features
        pycnidiation_density_features = {
            "mean_p_density": np.mean(density_array),
            "variance_p_density": np.var(density_array),
            "min_p_density": np.min(density_array),
            "max_p_density": np.max(density_array),
            "median_p_density": np.median(density_array)
        }

        features = distance_features | lesion_density_features | pycnidiation_density_features

    return features, pycn_contour

