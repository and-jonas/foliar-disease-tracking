
# ======================================================================================================================
# Various helper functions
# ======================================================================================================================

import numpy as np
import cv2
from scipy.spatial import KDTree
from scipy.spatial import distance as dist
from skimage.feature import peak_local_max
from PIL import Image
import skimage
import copy
# import matplotlib
# import matplotlib.pyplot as plt
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


def separate_marks(pts):
    # Use polynomial to separate top and bottom marks
    coefficients = np.polyfit(pts[:, 0], pts[:, 1], deg=2)
    y_predicted = np.polyval(coefficients, pts[:, 0])
    residuals = pts[:, 1] - y_predicted

    # Find top and bottom points, using the residuals
    top_idx = np.where(residuals < 0)[0]
    bottom_idx = np.where(residuals > 0)[0]

    top = pts[top_idx]
    bottom = pts[bottom_idx]

    return top, bottom


# split top and bottom marks via clustering
def identify_outliers_2d(data, m):
    """
    Separates top and bottom points and performs filtering within each group based on y-coordinates
    :param data: the y-coordinates of all detected key-points in an image
    :param m: the number of sds to tolerate
    :return: the separated top and bottom keypoints, cleaned from outliers
    """

    # Use polynomial to separate top and bottom marks
    coefficients = np.polyfit(data[:, 0], data[:, 1], deg=3)
    y_predicted = np.polyval(coefficients, data[:, 0])
    residuals = data[:, 1] - y_predicted

    # Find top and bottom points, using the residuals
    top_idx = np.where(residuals < 0)[0]
    bottom_idx = np.where(residuals > 0)[0]

    # find top and bottom outliers
    bottom_y = data[bottom_idx, 1]
    bottom_outliers = reject_outliers(data=bottom_y, tol=None, m=m)
    top_y = data[top_idx, 1]
    top_outliers = reject_outliers(data=top_y, tol=None, m=m)

    outliers = top_outliers + bottom_outliers

    # clean by removing detected outliers
    bottom_idx_clean = np.delete(bottom_idx, bottom_outliers, 0)
    top_idx_clean = np.delete(top_idx, top_outliers, 0)
    kpts_bottom = data[bottom_idx_clean, :]
    kpts_top = data[top_idx_clean, :]

    # sort from left to right
    kpts_bottom = kpts_bottom[np.argsort(kpts_bottom[:, 0]), :]
    kpts_top = kpts_top[np.argsort(kpts_top[:, 0]), :]

    return kpts_top, kpts_bottom, outliers



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
    if np.abs(current_size-mean_size_prev) > max_diff:
        idx = [len(data)-1]
    else:
        idx = []
    return idx


def remove_double_detections(x, y):
    """
    Removes one of two coordinate pairs if their distance is below 50
    :param x:
    :param y:
    :return:
    """
    point_list = np.array([[a, b] for a, b in zip(x, y)], dtype=np.int32)
    dist_mat = dist.cdist(point_list, point_list, "euclidean")
    np.fill_diagonal(dist_mat, np.nan)
    dbl_idx = np.where(dist_mat < 50)[0].tolist()[::2]
    point_list = np.delete(point_list, dbl_idx, axis=0)
    x = np.delete(x, dbl_idx, axis=0)
    y = np.delete(y, dbl_idx, axis=0)
    return point_list, x, y


def make_bbox_overlay(img, points_top, points_bottom, box):
    """
    Creates an overlay on the original image that shows the detected marks and the fitted bounding box
    :param img: original image
    :param point_list: list of coordinate [x,y] coordinate pairs denoting the detected mark positions
    :param box: the box coordinates in cv2 format
    :return:
    """
    overlay = copy.copy(img)
    for point in points_top:
        cv2.circle(overlay, (point[0], point[1]), radius=15, color=(0, 0, 255), thickness=9)
    for point in points_bottom:
        cv2.circle(overlay, (point[0], point[1]), radius=15, color=(255, 255, 0), thickness=9)
    box_ = np.intp(box)
    cv2.drawContours(overlay, [box_], 0, (255, 0, 0), 9)
    overlay = cv2.resize(overlay, (0, 0), fx=0.25, fy=0.25)
    return overlay


def make_inference_crop(pts, img,):
    # Calculate the centroid
    mw, mh = np.mean(pts, axis=0)

    # Crop According to the centroid
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
    Removes the predicted pycnidia and rust pustules from the mask. Replaces the relevant pixel values with the average
    of the surrounding pixels. Points need to be transformed separately and added again to the transformed mask.
    :param mask: the mask to remove the points from
    :param points: to coordinates of the points to be removed
    :return: The "cleaned" mask
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


def rotate_translate_warp_points(points, rot, box, mat_proj, mat_pw, target_shape):
    """
    rotates, translates, and warps point coordinates to match the transformed mask.
    Filters detected point lying outside the roi.
    :param points: a list of 2D points
    :param rot: rotation matrix applied to the msak
    :param box: the corner coordinates of the bounding box used to crop he roi from the image
    :param warp: the transformation matrix
    :param target_shape: the dimension of the desired output image
    :return: transformed point coordinates
    """
    # rotate
    points_rot = np.intp(cv2.transform(np.array([points]), rot))[0]

    # translate
    tx, ty = (-box[0][0], -box[0][1])
    translation_matrix = np.array([
        [1, 0, tx],
        [0, 1, ty]
    ], dtype=np.float32)
    points_trans = np.intp(cv2.transform(np.array([points_rot]), translation_matrix))[0]

    if mat_proj is not None:
        # make an AffineTransform object for direct application to points
        tform = skimage.transform.AffineTransform(matrix=mat_proj)
        # TODO is the inverse needed??? Is that a scikit-image vs OpenCV issue??
        tf_points_proj = tform.inverse(points_trans).astype("int64")
    else:
        tf_points_proj = points_trans
    if mat_pw is not None:
        tform = mat_pw
        tf_points_pw = tform.inverse(points_trans).astype("uint64")
    else:
        tf_points_pw = points_trans

    # remove any point outside the roi
    # Create a boolean mask based on the ROI
    mask_proj = (tf_points_proj[:, 1] < target_shape[0]) & (tf_points_proj[:, 1] > 0) & \
                (tf_points_proj[:, 0] < target_shape[1]) & (tf_points_proj[:, 0] > 0)
    mask_pw = (tf_points_pw[:, 1] < target_shape[0]) & (tf_points_pw[:, 1] > 0) & \
              (tf_points_pw[:, 0] < target_shape[1]) & (tf_points_pw[:, 0] > 0)

    # Filter points based on the mask
    filtered_points_proj = tf_points_proj[mask_proj]
    filtered_points_pw = tf_points_pw[mask_pw]

    return [filtered_points_proj, filtered_points_pw]


def rotate_translate_warp_points2(mask, classes, rot, box, tf, target_shape, warped):
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


def add_points_to_mask(mask, pycn_trf, rust_trf):
    """
    Adds the transformed detected pycnidia and rust pustules to the transformed mask
    :param mask: transformed mask without pycnidia and rust
    :param pycn_trf: transformed coordinates of the pycnidia
    :param rust_trf: transformed coordinates of the rust pustules
    :return: the complemented transformed mask
    """
    # add points again
    try:
        mask[pycn_trf[:, 1], pycn_trf[:, 0]] = 4  # pycnidia
    except TypeError:
        pass
    try:
        mask[rust_trf[:, 1], rust_trf[:, 0]] = 5  # rust
    except TypeError:
        pass

    # to ease inspection
    mask = (mask.astype("uint32")) * 255 / 5
    mask = mask.astype("uint8")

    return mask


def find_keypoint_matches(current, current_orig, ref, dist_limit=150):
    """
    Finds pairs of matching detected marks on two subsequent images of a series
    :param current: the current image to be aligned to the initial image
    :param current_orig: the initial image of the series
    :param ref:
    :param dist_limit: the maximum allowed distance between points to consider them the same point
    :return:
    """
    # MAKE AND QUERY TREE
    tree = KDTree(current)
    assoc = []
    for I1, point in enumerate(ref):
        _, I2 = tree.query(point, k=1, distance_upper_bound=dist_limit)
        assoc.append((I1, I2))
    # match indices back to key point coordinates
    assocs = []
    for a in assoc:
        p1 = ref[a[0]].tolist()
        try:
            p2 = current_orig[a[1]].tolist()
        except IndexError:
            p2 = [np.NAN, np.NAN]
        assocs.append([p1, p2])

    # reshape to list of corresponding source and target key point coordinates
    pair = assocs
    src = [[*p[0]] for p in pair if p[1][0] is not np.nan]
    dst = [[*p[1]] for p in pair if p[1][0] is not np.nan]

    return src, dst


def check_keypoint_matches(src, dst, mdev, tol, m):
    """
    Verifies that the kd-tree identified associations are meaningful by comparing the distance between source and target
    across the top and bottom rows. Regular patterns are expected, and outliers from this pattern are removed.
    If no stable pattern is found, all associations are deleted.
    :param src: source point coordinates
    :param dst: destination point coordinates
    :param mdev: average deviation from mean that is tolerated for associations
    :param m: parameter for outlier removal
    :return: cleaned lists of source and destination points
    """

    if len(src) < 7:
        src, dst = [], []
    else:
        # broadly check for a regular pattern, if none is found delete all associations
        distances = pairwise_distances(src, dst)
        d = np.abs(distances - np.mean(distances))
        m_dev = np.mean(d)
        if mdev is not None and m_dev > mdev:
           src, dst = [], []
        # otherwise, separately evaluate pairwise distances for top and bottom marks
        # eliminate outliers from both, source and target, if any found
        else:
            outliers = []
            for type in [src, dst]:
                top, bottom = separate_marks(pts=np.array(type))
                top_distances = distances[:len(top)]
                bottom_distances = distances[len(bottom):]
                outliers_top = reject_outliers(data=top_distances, tol=tol, m=m)
                outliers_bottom = reject_outliers(data=bottom_distances, tol=tol, m=m)
                outliers_bottom = [i + len(top) for i in outliers_bottom]
                outliers.extend(outliers_top + outliers_bottom)

            src = np.delete(src, outliers, 0)
            dst = np.delete(dst, outliers, 0)

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


def expand_bbox_to_image_edge(pts, img):
    """
    takes for coordinate pairs of the rotated bounding box
    and expands it to the edge of the image
    :param pts: four coordinate pairs representing tl, tr, bl, br as an an array
    :param img: the rotated image
    :return: the corner coordinates of the expanded bounding box
    """
    [tl, tr, bl, br] = pts
    tl = [0, tl[1]]
    tr = [img.shape[1], tr[1]]
    bl = [0, bl[1]]
    br = [img.shape[1], br[1]]
    return np.asarray([tl, tr, bl, br])


def make_point_list(input):
    """
    Transform cv2 format to ordinary point list
    :param input:
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


def make_overlay(patch, mask, colors=[(1, 0, 0, 0.25)]):
    img_ = Image.fromarray(patch, mode="RGB")
    img_ = img_.convert("RGBA")
    class_labels = np.unique(mask)
    for i, v in enumerate(class_labels[1:]):
        r, g, b, a = colors[i]
        M = np.where(mask == v, 255, 0)
        M = M.ravel()
        M = np.expand_dims(M, -1)
        out_mask = np.dot(M, np.array([[r, g, b, a]]))
        out_mask = np.reshape(out_mask, newshape=(patch.shape[0], patch.shape[1], 4))
        out_mask = out_mask.astype("uint8")
        M = Image.fromarray(out_mask, mode="RGBA")
        img_.paste(M, (0, 0), M)
    img_ = img_.convert('RGB')
    overlay = np.asarray(img_)

    return overlay


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

