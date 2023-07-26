import numpy as np
import cv2
import math
from scipy.spatial.distance import cdist
from scipy.spatial import distance as dist


def order_points(pts):
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
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="int")


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