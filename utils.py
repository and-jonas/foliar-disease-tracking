
# ======================================================================================================================
# Various helper functions
# ======================================================================================================================

import numpy as np
import cv2
import math
from scipy.spatial import KDTree
from scipy.spatial import distance as dist
from scipy.spatial.distance import cdist
from PIL import Image
import skimage
from scipy.ndimage import map_coordinates
import copy
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('Qt5Agg')


def reject_outliers(data, m=2.):
    """
    Detects outliers in 1d and returns the list index of the outliers
    :param data: 1d array
    :param m:
    :return: list index of outliers
    """
    d = np.abs(data - np.mean(data))
    mdev = np.mean(d)
    s = d / mdev if mdev else np.zeros(len(d))
    idx = np.where(s > m)[0].tolist()
    return idx


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
    Removes one of two coordinate pairs if theIR distance is below 75
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


def make_bbox_overlay(img, point_list, box):
    overlay = copy.copy(img)
    for point in point_list:
        cv2.circle(overlay, (point[0], point[1]), radius=15, color=(0, 0, 255), thickness=9)
    box_ = np.intp(box)
    cv2.drawContours(overlay, [box_], 0, (255, 0, 0), 9)
    overlay = cv2.resize(overlay, (0, 0), fx=0.25, fy=0.25)
    return overlay


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


def remove_points_from_mask(mask, points):
    """
    Removes the predicted pycnidia and rust pustules from the mask. Replaces the relevant pixel values with the average
    of the surrounding pixels. Points need to be transformed separately and added again to the transformed mask.
    :param mask: the mask to remove the points from
    :param points: to coordinates of the points to be removed
    :return: The "cleaned" mask
    """
    y_points, x_points = points
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
    Fitlters detected point lying outside the roi.
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


def add_points_to_mask(mask, pycn_trf, rust_trf):

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


def getComponents(normalised_homography):
  '''((translationx, translationy), rotation, (scalex, scaley), shear)'''
  a = normalised_homography[0,0]
  b = normalised_homography[0,1]
  c = normalised_homography[0,2]
  d = normalised_homography[1,0]
  e = normalised_homography[1,1]
  f = normalised_homography[1,2]

  p = math.sqrt(a*a + b*b)
  r = (a*e - b*d)/(p)
  q = (a*d+b*e)/(a*e - b*d)

  translation = (c,f)
  scale = (p,r)
  shear = q
  theta = math.atan2(b,a)

  return translation, theta, scale, shear


def order_points(pts):
    """
    Orders a list of points clock-wise
    :param pts: List of point coordinates pairs as [x, y]
    :return: the coordinates of the top-left, top-right, bottom-right, and bottom-left points
    """
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[-2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
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


def make_point_list_(input):
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
        point_list.append([a, b])
    c = point_list

    return c


def flatten_centroid_data(input, asarray):
    xs = []
    ys = []
    for point in input:
        x = point[0]
        y = point[1]
        xs.append(x)
        ys.append(y)
    point_list = []
    for a, b in zip(xs, ys):
        point_list.append([a, b])
        c = point_list
    if asarray:
        c = np.asarray(point_list)
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


def sort_counterclockwise(points, start, centre=None):
    if centre:
        centre_x, centre_y = centre
    else:
        centre_x, centre_y = sum([x for x, _ in points]) / len(points), sum([y for _, y in points]) / len(points)
    angles = [math.atan2(y - centre_y, x - centre_x) for x, y in points]
    vector = [(x-centre_x, y-centre_y) for x, y in points]
    # Length of vector: ||v||
    lenvector = [math.hypot(x, y) for x, y in vector]
    indices = np.argsort(angles)
    angles = [angles[i] for i in indices]
    lengths = [lenvector[i] for i in indices]
    counterclockwise_points = [points[i] for i in indices]
    pp = np.stack(counterclockwise_points)
    start_index = [any(ele == start) for ele in pp]
    start_index = np.where(start_index)[0][0]
    final = counterclockwise_points[start_index:] + counterclockwise_points[:start_index]
    angles = angles[start_index:] + angles[:start_index]
    lengths = lengths[start_index:] + lengths[:start_index]
    return indices, angles, lengths, final


def find_top_left(pts):
    # identify all left-most points
    sorted = pts[pts[:, 0].argsort()]
    diffs_x = np.diff(sorted, axis=0)[:, 0]
    index = np.min(np.where(diffs_x > 150))
    left_points = sorted[:index+1, :]
    ind_tl = np.argsort(left_points[:, 1])[0]
    top_left = sorted[ind_tl]
    # identify all right-most points
    sorted_inv = sorted[::-1]
    diffs_x = np.diff(sorted_inv, axis=0)[:, 0]
    index = np.min(np.where(diffs_x < -150))
    right_points = sorted_inv[:index+1, :]
    ind_tr = np.argsort(right_points[:, 1])[0]
    top_right = sorted_inv[ind_tr]

    return top_left, top_right


def distance(P1, P2):
    return ((P1[0] - P2[0])**2 + (3*(P1[1] - P2[1]))**2) ** 0.5


def optimized_path(coords, start=None):
    if start is None:
        start = coords[0]
    pass_by = coords
    path = [start]
    pass_by.remove(start)
    while pass_by:
        nearest = min(pass_by, key=lambda x: distance(path[-1], x))
        path.append(nearest)
        pass_by.remove(nearest)
    return path


def find_marker_centroids(image, size_th, coordinates, leaf_mask):
    """
    Filter objects in a binary mask by size
    :param mask: A binary mask to filter
    :param size_th: The size threshold used to filter (objects GREATER than the threshold will be kept)
    :return: A binary mask containing only objects greater than the specified threshold
    """

    # threshold
    lower = np.array([140, 140, 140])  # v changed successively from 35 to 30 to 22 for selected images
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(image, lower, upper)
    mask = cv2.medianBlur(mask, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    mask = cv2.dilate(mask, kernel)
    # remove everything outside the leaf
    mask = np.bitwise_and(mask, leaf_mask)
    # detect and filter based on size
    _, output, stats, ctr = cv2.connectedComponentsWithStats(mask, connectivity=4)
    sizes = stats[1:, -1]
    idx = (np.where(sizes > size_th)[0] + 1).tolist()
    out = np.in1d(output, idx).reshape(output.shape)
    out = np.where(out, 255, 0).astype("uint8")
    # detect and filter based overlap with local maxima
    _, output, stats, ctr = cv2.connectedComponentsWithStats(out, connectivity=4)
    keep = [output[p[0], p[1]] for p in coordinates]
    keep = np.unique(keep)
    keep = np.delete(keep, np.where(keep == 0))
    mask2 = np.in1d(output, keep).reshape(output.shape)
    mask2 = np.where(mask2, 255, 0).astype("uint8")
    _, output, stats, ctr = cv2.connectedComponentsWithStats(mask2, connectivity=4)
    ctr = ctr[1:]
    ctr = ctr.astype("int")

    upper_left, _ = find_top_left(pts=ctr)

    list_ = make_point_list_(ctr)
    path = optimized_path(coords=list_, start=[upper_left[0], upper_left[1]])

    # indices, angles, lengths, sorted_ = sort_counterclockwise(list_, start=upper_left)
    # diffs = np.diff(lengths).astype("int")
    # diffdiff = np.diff(diffs)
    #
    # errors = np.where(np.absolute(diffdiff) > 3*np.mean(np.absolute(diffdiff)))
    #
    # errors = np.concatenate(errors).tolist()
    # errors = [e+1 for e in errors]
    # for e in errors:
    #     if np.absolute(angles[e] - angles[e+1]) < 0.05:
    #         sorted_[e], sorted_[e+1] = sorted_[e+1], sorted_[e]
    #         lengths[e], lengths[e+1] = lengths[e+1], lengths[e]
    #         diffs = np.diff(lengths).astype("int")
    #         diffdiff = np.diff(diffs)
    #     if any(diffdiff) > 3*np.mean(np.absolute(diffdiff)):
    #         continue
    #     else:
    #         break

    # ctr_reord = np.stack([ctr[i] for i in indices], axis=0)
    ctr_reord = path

    # ctr = ctr.astype("int")
    cleaned = np.zeros_like(mask)
    cleaned = np.where(output == 0, cleaned, 255)
    for i, point in enumerate(path):
        # cv2.putText(cleaned,
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #             org=point,
        #             text=str(i),
        #             thickness=5,
        #             fontScale=3,
        #             color=122)
        cv2.putText(image,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    org=point,
                    text=str(i),
                    thickness=5,
                    fontScale=3,
                    color=(255, 0, 0))
    return cleaned, ctr_reord, image


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


def keep_central_object(mask):
    """
    Filter objects in a binary mask by centroid position
    :param mask: A binary mask to filter
    :return: A binary mask containing only the central object (by centroid position)
    """
    n_comps, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    ctr_img = centroids[0:1]
    dist = cdist(centroids[1:], ctr_img)
    min_idx = np.argmin(dist)
    lesion_mask = np.uint8(np.where(output == min_idx + 1, 255, 0))

    return lesion_mask


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


def interpolate_transformed_image_rgb(image, tform, output_shape):

    # Generate grid of coordinates for the output image
    grid_x, grid_y = np.meshgrid(np.arange(output_shape[0]), np.arange(output_shape[1]))
    output_coords = np.vstack([grid_x.ravel(), grid_y.ravel()])

    # Apply the inverse transform to get coordinates in the original image space
    src_coords = tform.inverse(output_coords.T).T

    # Perform interpolation using map_coordinates for each channel
    channels = [map_coordinates(image[:, :, i], src_coords, order=3, mode='constant').reshape(output_shape) for i in range(image.shape[2])]

    # Stack the channels to form the RGB image
    return np.stack(channels, axis=-1)


def Intersect2Circles(A, a, B, b):
    # A, B = [x, y]
    # return = [Q1, Q2] or [Q] or [] where Q = [x, y]
    AB0 = B[0] - A[0]
    AB1 = B[1] - A[1]
    c = math.sqrt(AB0 * AB0 + AB1 * AB1)

    if c == 0:
        # no distance between centers
        return []

    x = (a * a + c * c - b * b) / (2 * c)
    y = a * a - x * x

    if y < 0:
        # no intersection
        return []

    if y > 0:
        y = math.sqrt(y)

    # compute unit vectors ex and ey
    ex0 = AB0 / c
    ex1 = AB1 / c
    ey0 = -ex1
    ey1 = ex0

    Q1x = A[0] + x * ex0
    Q1y = A[1] + x * ex1

    if y == 0:
        # one touch point
        return [[Q1x, Q1y]]

    # two intersections
    Q2x = Q1x - y * ey0
    Q2y = Q1y - y * ey1
    Q1x += y * ey0
    Q1y += y * ey1

    return [[Q1x, Q1y], [Q2x, Q2y]]


def get_corners(pts):

    upper = pts[pts[:, 1] < 300]
    lower = pts[pts[:, 1] > 300]

    # sort the points based on their x-coordinates
    upper_x_sort = upper[np.argsort(upper[:, 0]), :]
    lower_x_sort = lower[np.argsort(lower[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    upper_left = upper_x_sort[:1, :][0]
    upper_right = upper_x_sort[-1:, :][0]
    lower_left = lower_x_sort[:1, :][0]
    lower_right = lower_x_sort[-1:, :][0]

    return np.array([upper_left, upper_right, lower_left, lower_right], dtype="int")




