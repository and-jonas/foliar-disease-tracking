
# ======================================================================================================================
# Various helper functions
# ======================================================================================================================

import numpy as np
import cv2
import math
from scipy.spatial import KDTree
from scipy.spatial import distance as dist


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
    :param d: size difference threshold in px
    :param data: 1d array
    :param m:
    :return: list index of outliers
    """
    mean_size_prev = np.mean(data[:-1])
    current_size = data[-1]
    if np.abs(current_size-mean_size_prev) > max_diff:
        idx = [len(data)-1]
    else:
        idx = []
    return idx


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


def find_keypoint_matches(current, current_orig, ref, dist_limit=150):
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
    rightMost = xSorted[2:, :]
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