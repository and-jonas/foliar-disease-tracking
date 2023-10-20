
import math
import scipy.interpolate as si
from skimage.draw import line
import cv2
import numpy as np
import copy
import utils
from scipy import ndimage as ndi
import imutils
from skimage.segmentation import watershed


def reject_borders(image_):
    """
    Removes objects that touch on image borders
    :param image_: the binary image on which to perform the cleaning operation
    :return: the cleaned binary image
    """
    out_image = image_.copy()
    h, w = image_.shape[:2]
    for row in range(h):
        if out_image[row, 0] == 255:
            cv2.floodFill(out_image, None, (0, row), 0)
        if out_image[row, w - 1] == 255:
            cv2.floodFill(out_image, None, (w - 1, row), 0)
    for col in range(w):
        if out_image[0, col] == 255:
            cv2.floodFill(out_image, None, (col, 0), 0)
        if out_image[h - 1, col] == 255:
            cv2.floodFill(out_image, None, (col, h - 1), 0)
    return out_image


def get_bounding_boxes(rect):
    """
    Get bounding boxes of each maintained lesion in a full leaf image
    :param mask: Binary segmentation mask of the image to process
    :param check_img: A copy of the corresponding image
    :return: Coordinates of the bounding boxes as returned by cv2.boundingRect()
    """
    x, y, w, h = rect
    w = w + 30
    h = h + 30
    x = x - 15
    y = y - 15
    # boxes must not extend beyond the edges of the image
    if x < 0:
        w = w-np.abs(x)
        x = 0
    if y < 0:
        h = h-np.abs(y)
        y = 0
    coords = x, y, w, h

    return coords


def select_roi(rect, img, mask):
    """
    Selects part of an image defined by bounding box coordinates. The selected patch is pasted onto empty masks for
    processing in correct spatial context
    :param rect: bounding box coordinates (x, y, w, h) as returned by cv2.boundingRect()
    :param img: The image to process
    :param mask: The binary mask of the same image
    :return: Roi of masks and img, centroid coordinates of the lesion to process (required for clustering)
    """
    # get the coordinates of the rectangle to process
    x, y, w, h = rect

    # create empty files for processing in spatial context
    empty_img = np.ones(img.shape).astype('int8') * 255
    empty_mask = np.zeros(mask.shape)
    empty_mask_all = np.zeros(mask.shape)

    # filter out the central object (i.e. lesion of interest)
    # isolate the rectangle
    patch_mask_all = mask[y:y + h, x:x + w]
    # # remove objects that touch on the border
    # patch_mask_obj = reject_borders(patch_mask_all)

    # select object by size or by centroid position in the patch
    n_comps, output, stats, centroids = cv2.connectedComponentsWithStats(patch_mask_all, connectivity=8)

    # if there is more then one object in the roi, need to select the one of interest
    if n_comps > 2:
        sizes = list(stats[:, 4][1:])
        max_idx = np.argmax(sizes)
        lesion_mask = np.uint8(np.where(output == max_idx + 1, 255, 0))
        ctr_obj = [centroids[max_idx + 1][0] + x, centroids[max_idx + 1][1] + y]

    # if there is only one object, select it
    else:
        lesion_mask = np.uint8(np.where(output == 1, 255, 0))
        ctr_obj = [centroids[1][0] + x, centroids[1][1] + y]

    # paste the patches onto the empty files at the correct position
    empty_img[y:y + h, x:x + w, :] = img[y:y + h, x:x + w, :]
    empty_mask[y:y + h, x:x + w] = lesion_mask
    empty_mask_all[y:y + h, x:x + w] = patch_mask_all

    mask_all = empty_mask_all.astype("uint8")
    mask = empty_mask.astype("uint8")

    return mask_all, mask, empty_img, ctr_obj


def check_color_profiles(color_profiles, dist_profiles_outer, spline_normals, remove=False):
    """
    Removes spline normals (and corresponding color profiles) that (a) extend into the lesion sphere of the same lesion
    (convexity defects) and replaces values on the inner side of the spline normals that lie beyond the "center" of the
    lesion (i.e. extend too far inwards).
    :param color_profiles: A 3D array (an image), raw color profiles, sampled on the spline normals
    :param dist_profiles: the euclidian distance map
    :param dist_profiles_outer: the euclidian distance map of the inverse binary image
    :param spline_normals: the spline normals in cv2 format
    :param remove: Boolean, whether or not the spline "problematic" spline normals are removed.
    :return: Cleaned color profiles (after removing normals in convexity defects and replacing values of normals
    extending too far inwards) and the cleaned list of spline normals in cv2 format.
    """

    # (2) OUTWARDS =====================================================================================================

    dist_profiles_outer = dist_profiles_outer.astype("int32")
    diff_out = np.diff(dist_profiles_outer, axis=0)

    if remove:
        # remove problematic spline normals
        checker_out_shape = np.unique(np.where(diff_out < 0)[1]).tolist()
        checker_out = checker_out_shape
        checker_out = np.unique(checker_out)
        spline_normals_clean = [i for j, i in enumerate(spline_normals) if j not in checker_out]
        # remove corresponding color profiles
        # color_profiles_ = np.delete(color_profiles_, checker_out, 1)
        color_profiles_ = np.delete(color_profiles, checker_out, 1)
        return color_profiles_, spline_normals_clean

    else:
        # separate the normals into complete and incomplete
        cols_1 = np.where(diff_out < 0)[1]
        # incomplete because extending outside of the ROI
        # in extreme vases
        cols_2 = np.where(np.all(diff_out == 0, axis=0))[0]
        # starting with a low distance value, in intermediate cases
        cols_3 = np.where(dist_profiles_outer[dist_profiles_outer.shape[0]-1] < 5, )[0]
        try:
            cols = np.append(cols_1, cols_2, cols_3)
        except TypeError:
            cols = []
        spline_normals_fulllength = [i for j, i in enumerate(spline_normals) if j not in np.unique(cols)]
        spline_normals_redlength = [i for j, i in enumerate(spline_normals) if j in np.unique(cols)]

        # identify where profiles extend into the "sphere" of another near-by lesion
        ind = []
        for i in range(diff_out.shape[1]):
            result = np.where(diff_out[:, i] < 0)[0]
            if result.size > 0:
                cut_idx = np.min(np.where(diff_out[:, i] < 0))
            else:
                cut_idx = np.nan
            ind.append(cut_idx)

        # for all pixels above this intersection
        # replace pixels with white pixels
        for i in range(color_profiles.shape[1]):
            if ind[i] is not np.nan:
                color_profiles[ind[i]:, i] = (255, 255, 255)

        # replace all pixel values on normals extending outside of the roi (leaf)
        color_profiles[:, cols_2] = (255, 255, 255)

        return color_profiles, spline_normals_fulllength, spline_normals_redlength


def remove_neighbor_lesions(checked_profiles, dist_profiles_multi, spl_n_clean, remove=False):
    """
    Removes color profiles that extend into the "incluencing sphere" of close-by other lesions; ALTERNATIVELY,
    the limits of the "sphere of influence" of a lesion is extracted, and color profiles are maintained until this point,
    whereas pixels beyond this point are set to white.
    :param remove: Boolean; Wether "incomplete" color profiles should be removed.
    :param checked_profiles: A 3D array (an image). The color profiles returned by check_color_profiles
    :param dist_profiles_multi: Distance profiles, sampled spline normals (distance map on binary mask with all objects)
    :param spl_n_clean: The maintained spline normals in cv2 format
    :return: A subset of the input color profiles, where profiles exending into near-by lesions have been removed,
    The corresponding spline normals in cv2 format.
    """

    dist_profiles_multi_ = dist_profiles_multi.astype("int32")
    diff_out = np.diff(dist_profiles_multi_, axis=0)

    if remove:
        checker_out = np.unique(np.where(diff_out < 0)[1]).tolist()
        # remove problematic spline normals
        spline_normals_clean = [i for j, i in enumerate(spl_n_clean) if j not in checker_out]
        color_profiles_ = np.delete(checked_profiles, checker_out, 1)
        return color_profiles_, spline_normals_clean

    else:
        # separate the normals into complete and incomplete
        cols = np.where(diff_out < 0)[1]
        spline_normals_fulllength = [i for j, i in enumerate(spl_n_clean) if j not in np.unique(cols)]
        spline_normals_redlength = [i for j, i in enumerate(spl_n_clean) if j in np.unique(cols)]

        # identify where profiles extend into the "sphere" of another near-by lesion
        ind = []
        for i in range(diff_out.shape[1]):
            result = np.where(diff_out[:, i] < 0)[0]
            if result.size > 0:
                cut_idx = np.min(np.where(diff_out[:, i] < 0))
            else:
                cut_idx = np.nan
            ind.append(cut_idx)

        # for all pixels above this intersection
        # replace pixels with white pixels
        color_profiles_ = copy.copy(checked_profiles)
        for i in range(color_profiles_.shape[1]):
            if ind[i] is not np.nan:
                color_profiles_[ind[i]:, i] = (255, 255, 255)

        return color_profiles_, spline_normals_fulllength, spline_normals_redlength


def get_spline_normals(spline_points, length_in=35, length_out=40):
    """
    Gets spline normals (lines) in cv2 format
    :param spline_points: x- and y- coordinates of the spline base points, as returned by spline_approx_contour().
    :param length_in: A numeric, indicating how far splines should extend inwards
    :param length_out: A numeric, indicating how far splines should extend outwards
    :return: A list of the spline normals, each in cv2 format.
    """
    ps = np.vstack((spline_points[1], spline_points[0])).T

    x_i = spline_points[0]
    y_i = spline_points[1]

    # get endpoints of the normals
    endpoints = []
    for i in range(0, len(ps)-1):
        v_x = y_i[i] - y_i[i + 1]
        v_y = x_i[i] - x_i[i + 1]
        mag = math.sqrt(v_x * v_x + v_y * v_y)
        v_x = v_x / mag
        v_y = v_y / mag
        temp = v_x
        v_x = -v_y
        v_y = temp
        A_x = int(y_i[i] + v_x * length_in)
        A_y = int(x_i[i + 1] + v_y * length_in)
        B_x = int(y_i[i] - v_x * length_out)
        B_y = int(x_i[i + 1] - v_y * length_out)
        n = [A_x, A_y], [B_x, B_y]
        endpoints.append(n)

    # get normals (lines) connecting the endpoints
    normals = []
    for i in range(len(endpoints)):
        p1 = endpoints[i][0]
        p2 = endpoints[i][1]
        discrete_line = list(zip(*line(*p1, *p2)))
        discrete_line = [[list(ele)] for ele in discrete_line]
        cc = [np.array(discrete_line, dtype=np.int32)]  # cv2 contour format
        normals.append(cc)

    return normals


def extract_normals_pixel_values(img, normals, length_in, length_out):
    """
    Extracts the pixel values situated on the spline normals.
    :param img: The image, binary mask or edt to process
    :param normals: The normals extracted in cv2 format as resulting from utils.get_spline_normals()
    :return: The "scan", i.e. an image (binary, single-channel 8-bit, or RGB) with stacked extracted profiles
    """
    # check whether is multi-channel image or 2d array
    is_img = utils.is_multi_channel_img(img)

    # For a normal perfectly aligned with the image axes, length equals the number of inward and outward pixels defined
    # utils.get_spline_normals()
    # All normals (differing in "pixel-length" due to varying orientation in space, are interpolated to the same length
    max_length_contour = length_in + length_out + 1

    # iterate over normals
    profile_list = []
    for k, normal in enumerate(normals):

        # get contour pixel coordinates
        contour_points = utils.flatten_contour_data(normal, asarray=False)

        # extract pixel values
        values = []
        for i, point in enumerate(contour_points):

            x = point[1]
            y = point[0]

            # try to sample pixel values, continue if not possible (extending outside of the roi)
            try:
                value = img[x, y].tolist()
            except IndexError:
                continue
            values.append(value)

            # split channels (R,G,B)
            # if img is a 3d array:
            if len(img.shape) > 2:
                channels = []
                for channel in range(img.shape[2]):
                    channel = [item[channel] for item in values]
                    channels.append(channel)
            else:
                channels = [values]

        # interpolate pixel values on contours to ensure equal length of all contours
        # for each channel
        interpolated_contours = []
        for channel in channels:
            size = len(channel)
            xloc = np.arange(len(channel))
            new_size = max_length_contour
            new_xloc = np.linspace(0, size, new_size)
            new_data = np.interp(new_xloc, xloc, channel).tolist()
            interpolated_contours.extend(new_data)

        if is_img:
            # create list of arrays
            line_scan = np.zeros([max_length_contour, 1, 3], dtype=np.uint8)
            for i in range(max_length_contour):
                v = interpolated_contours[i::max_length_contour]
                line_scan[i, :] = v
        else:
            line_scan = np.zeros([max_length_contour, 1], dtype=np.uint8)
            for i in range(max_length_contour):
                v = interpolated_contours[i::max_length_contour]
                line_scan[i, :] = v

        profile_list.append(line_scan)

    # stack arrays
    scan = np.hstack(profile_list)

    return scan


def spline_approx_contour(contour, interval, task="smoothing"):
    """
    Approximates lesion edges by b-splines
    :param contour: Contours detected in a binary mask.
    :param interval: A numeric, indicating the distance between contour points to be kept for fitting the b-spline. A
    higher value means more aggressive smoothing.
    :param task: A character vector, either "smoothing" or "basepoints". Affects where the b-spline is evaluated. If
    "smoothing", it is evaluated so often as to obtain a continuous contour. If "basepoints", it is evaluated at the
    desired distances to obtain spline normal base points coordinates.
    :return: x and y coordinates of pixels making up the smoothed contour, OR representing the spline normal base
    points.
    """
    # re-sample contour points
    contour_points = utils.flatten_contour_data(contour, asarray=True)[::interval]
    # find B-Spline representation of contour
    tck, u = si.splprep(contour_points.T, u=None, s=0.0, per=1, quiet=1)
    # evaluate  B-spline
    if task == "smoothing":
        u_new = np.linspace(u.min(), u.max(), len(contour_points)*interval*2)
    elif task == "basepoints":
        u_new = np.linspace(u.min(), u.max(), int(len(contour_points)/2))
    y_new, x_new = si.splev(u_new, tck, der=0)
    # format output
    if task == "smoothing":
        x_new = x_new.astype("uint32")
        y_new = y_new.astype("uint32")
    return x_new, y_new


def spline_contours(mask_obj, mask_all, img, checker):
    """
    Wrapper function for processing of contours via spline normals
    :param mask_obj: a binary mask containing only the lesion of interest.
    !!IMPORTANT!!: This must be in uint8 (i.e. 0 and 255), otherwise ndi.distance_transform_edt() produces nonsense
    output !!
    :param mask_all: a binary mask containing all the segmented objects in the patch
    :param img: the original patch image
    :param checker: A copy of the (full) image to process.
    :return: cleaned color profiles from contour normals in cv2 format, an image for evaluation
    """

    checker_filtered = copy.copy(checker)

    mask_invert = np.bitwise_not(mask_obj)

    # calculate the euclidian distance transforms on the original and inverted masks
    distance_invert = ndi.distance_transform_edt(mask_invert)

    # get contour
    contour, _ = cv2.findContours(mask_obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # get spline points
    spline_points = spline_approx_contour(contour, interval=1, task="basepoints")

    # get spline normals
    spl_n = get_spline_normals(spline_points=spline_points, length_in=0, length_out=15)

    # sample normals on image and on false object mask
    color_profiles = extract_normals_pixel_values(img=img, normals=spl_n, length_in=0, length_out=15)

    # sample normals on distance maps
    dist_profiles_outer = extract_normals_pixel_values(img=distance_invert, normals=spl_n, length_in=0, length_out=15)

    # remove normals that extend into lesion or beyond lesion "center"
    checked_profiles, spl_n_full, spl_n_red = check_color_profiles(
        color_profiles=color_profiles,
        dist_profiles_outer=dist_profiles_outer,
        spline_normals=spl_n,
        remove=False
    )

    # remove normals extending into neighbor lesions
    mask_all_invert = np.bitwise_not(mask_all)
    distance_invert_all = ndi.distance_transform_edt(mask_all_invert)

    dist_profiles_multi = extract_normals_pixel_values(distance_invert_all, spl_n_full, length_in=0, length_out=15)

    final_profiles, spl_n_full_l, spl_n_red_l = remove_neighbor_lesions(
        checked_profiles=checked_profiles,
        dist_profiles_multi=dist_profiles_multi,
        spl_n_clean=spl_n_full,
        remove=False
    )

    # create the check image: only complete profiles
    for i in range(len(spl_n_full)):
        cv2.drawContours(checker_filtered, spl_n_full[i], -1, (255, 0, 0), 1)
    # add contour
    contours, _ = cv2.findContours(mask_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        cv2.drawContours(checker_filtered, c, -1, (0, 0, 255), 1)

    # create the check image
    for i in range(len(spl_n_full)):
        cv2.drawContours(checker, spl_n_full[i], -1, (255, 0, 0), 1)
    for i in range(len(spl_n_red_l)):
        cv2.drawContours(checker, spl_n_red_l[i], -1, (0, 255, 0), 1)
    for i in range(len(spl_n_red)):
        cv2.drawContours(checker, spl_n_red[i], -1, (0, 122, 0), 1)
    # add contour
    contours, _ = cv2.findContours(mask_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        cv2.drawContours(checker, c, -1, (0, 0, 255), 1)

    return final_profiles, (checker, checker_filtered), (spl_n, spl_n_full_l, spl_n_red_l, spl_n_red), spline_points


def get_object_watershed_labels(current_mask, markers):
    """
    Performs watershed segmentation of merged objects
    :param current_mask: the current frame in the time series (a binary mask)
    :param markers: labelled components from the previous frame
    :return: the current frame with separated objects
    """
    # binarize the post-processed mask
    mask = np.where(current_mask == 125, 255, current_mask)
    # invert the mask
    mask_inv = np.bitwise_not(mask)
    # calculate the euclidian distance transform
    distance = ndi.distance_transform_edt(mask_inv)
    # watershed segmentation, using labelled components as markers
    # this must be done to ensure equal number of watershed segments and connected components!
    labels = watershed(distance, markers=markers, watershed_line=True)
    # make thicker watershed lines for separatbility of objects
    watershed_lines = np.ones(shape=np.shape(labels))
    watershed_lines[labels == 0] = 0  # ws lines are labeled as 0 in markers
    kernel = np.ones((2, 2), np.uint8)
    watershed_lines = cv2.erode(watershed_lines, kernel, iterations=1)
    labels = labels * watershed_lines
    labels = labels * current_mask/255
    separated = np.where(labels != 0, 255, 0)
    return np.uint8(separated)

