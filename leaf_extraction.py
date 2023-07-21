import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import utils
import glob
import os
import copy
import scipy.spatial
import skimage
matplotlib.use('Qt5Agg')

path_labels = "Z:/Public/Jonas/Data/ESWW007/SingleLeaf/Test/runs/pose/predict4/labels"
path_images = "Z:/Public/Jonas/Data/ESWW007/SingleLeaf/Test"

labels = glob.glob(f'{path_labels}/*.txt')
names = [os.path.basename(l).replace(".txt", "") for l in labels]

points_lists = []
for j, n in enumerate(names):

    # get key point coordinates
    coords = pd.read_table(f'{path_labels}/{n}.txt', header=None, sep=" ")
    x = coords.iloc[:, 5] * 8192
    y = coords.iloc[:, 6] * 5464

    # get image
    img = Image.open(f'{path_images}/{n}.JPG')
    img = np.array(img)
    # img = cv2.copyMakeBorder(img, 500, 500, 500, 500, cv2.BORDER_CONSTANT)

    # get minimum area rectangle around key points
    point_list = np.array([[a, b] for a, b in zip(x, y)], dtype=np.int32)
    rect = cv2.minAreaRect(point_list)

    # TODO enlarge bbox
    (center, (w, h), angle) = rect
    if angle > 45:
        angle = angle - 90

    # rotate the image about its center
    rows, cols = img.shape[0], img.shape[1]
    M_img = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    img_rot = cv2.warpAffine(img, M_img, (cols, rows))
    # plt.imshow(img_rot)

    # rotate the bbox about its center
    M_box = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    box = cv2.boxPoints(rect)
    pts = np.intp(cv2.transform(np.array([box]), M_box))[0]
    pts[pts < 0] = 0

    # order bbox points clockwise
    pts = utils.order_points(pts)

    # crop the roi from the rotated image
    img_crop = img_rot[pts[0][1]:pts[2][1], pts[0][0]:pts[1][0]]
    # plt.imshow(img_crop)
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

    # draw key points on overlay image as check
    for i, point in enumerate(kpts):
        cv2.circle(img_crop, (point[0], point[1]), radius=7, color=(0, 0, 255), thickness=-1)
        cv2.putText(img_crop,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    org=point,
                    text=str(i),
                    thickness=4,
                    fontScale=2,
                    color=(255, 0, 0))
    # plt.imshow(img_crop)

    # save output
    scale_factor = 0.5
    width = int(img_crop.shape[1] * scale_factor)
    height = int(img_crop.shape[0] * scale_factor)
    dim = (width, height)
    resized = cv2.resize(img_crop, dim, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(f'{path_images}/output/overlay/{n}.JPG', cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    # cv2.imwrite(f'{path_images}/output/crops/{n}.png', cv2.cvtColor(save_img, cv2.COLOR_BGR2RGB), [cv2.IMWRITE_PNG_COMPRESSION, 0])

    # match all images in the series to the first image
    if j == 0:
        kpts_ref = kpts
    elif j > 0:
        # match key points with those on the first image of the series
        # by searching for the closest points
        # if non is found in proximity, eliminate from both images
        tree = scipy.spatial.KDTree(kpts)
        assoc = []
        for I1, point in enumerate(kpts_ref):
            _, I2 = tree.query(point, k=1, distance_upper_bound=100)
            assoc.append((I1, I2))
        # match indices back to key point coordinates
        assocs = []
        for a in assoc:
            p1 = kpts_ref[a[0]].tolist()
            try:
                p2 = kpts[a[1]].tolist()
            except IndexError:
                p2 = [np.NAN, np.NAN]
            assocs.append([p1, p2])

        # reshape to list of corresponding source and target key point coordinates
        pair = assocs
        src = [[*p[0]] for p in pair if p[1][0] is not np.nan]
        dst = [[*p[1]] for p in pair if p[1][0] is not np.nan]

        # perspective Transform with the first image of the series as destination
        tform3 = skimage.transform.ProjectiveTransform()
        tform3.estimate(src, dst)
        warped = skimage.transform.warp(save_img, tform3)
        warped = skimage.util.img_as_ubyte(warped)
        # cv2.imwrite(f'{path_images}/output/warped/{n}.png', cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), [cv2.IMWRITE_PNG_COMPRESSION, 0])
        cv2.imwrite(f'{path_images}/output/warped/{n}.JPG', cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))


# points_lists.append(kpts)
#
#
# pairs = []
# for point_list in points_lists[1:]:
#     # match key points with those on the first image of the series
#     tree = scipy.spatial.KDTree(point_list)
#     assoc = []
#     for I1, point in enumerate(points_lists[0]):
#         _, I2 = tree.query(point, k=1, distance_upper_bound=100)
#         assoc.append((I1, I2))
#     # match indices back to kpt coordinates
#     assocs = []
#     for a in assoc:
#         p1 = tuple(points_lists[0][a[0]])
#         try:
#             p2 = tuple(point_list[a[1]])
#         except IndexError:
#             p2 = (np.NAN, np.NAN)
#         assocs.append([p1, p2])
#     pairs.append(assocs)
#
# pair = pairs[0]
#
# src = [[*p[0]] for p in pair if p[1][0] is not np.nan]
# dst = [[*p[1]] for p in pair if p[1][0] is not np.nan]
#
# tform3 = transform.ProjectiveTransform()
# tform3.estimate(src, dst)
#
# # get image
# img0 = Image.open(f'{path_images}/{names[0]}.JPG')
# img0 = np.array(img0)
# img1 = Image.open(f'{path_images}/{names[1]}.JPG')
# img1 = np.array(img1)
# warped = transform.warp(img1, tform3)
#
# plt.imshow(warped)
#
# fig, ax = plt.subplots(nrows=2, figsize=(8, 3))
#
# ax[0].imshow(img0, cmap=plt.cm.gray)
# # ax[0].plot(dst[:, 0], dst[:, 1], '.r')
# ax[1].imshow(warped, cmap=plt.cm.gray)
#
# for a in ax:
#     a.axis('off')
#
# plt.tight_layout()
# plt.show()
#
#
#
#
# box = cv2.boxPoints(rect)  # cv2.cv.BoxPoints(rect) for OpenCV <3.x
# box = np.intp(box)
#
#
# cv2.drawContours(img, [box], 0, (255, 0, 0), 5)
# plt.imshow(img)
#
# # rotate img
# angle = rect[2]
# rows,cols = img.shape[0], img.shape[1]
# M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
# img_rot = cv2.warpAffine(img,M,(cols,rows))
# plt.imshow(img_rot)
#
# # rotate bounding box
# rect0 = (rect[0], rect[1], 0.0)
# M = cv2.getRotationMatrix2D((int(rect[0][0]), int(rect[0][1])), angle, 1)
# box = cv2.boxPoints(rect0)
# pts = np.intp(cv2.transform(np.array([box]), M))[0]
# pts[pts < 0] = 0
#
# # crop
# img_crop = img_rot[pts[1][1]:pts[0][1],
#                    pts[1][0]:pts[2][0]]
# plt.imshow(img_crop)
#
# cropped = crop_minAreaRect(img, rect)
#
# plt.imshow(cropped)
#
# for i, point in enumerate(point_list):
#     cv2.putText(img,
#                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                 org=point,
#                 text=str(i),
#                 thickness=5,
#                 fontScale=3,
#                 color=(255, 0, 0))
# # plt.imshow(img)
#
# points_lists.append(point_list)
#
# tree = scipy.spatial.KDTree(points_lists[1])
# assoc = []
# for I1, point in enumerate(points_lists[0]):
# _, I2 = tree.query(point, k=1)
# assoc.append((I1, I2))
#
# for i, point in enumerate(points_lists[0]):
# cv2.putText(img,
#             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#             org=point,
#             text=str(i),
#             thickness=5,
#             fontScale=3,
#             color=(255, 0, 0))
# plt.imshow(img)
#
#
#
#
# upper_left, _ = utils.find_top_left(pts=point_list)
# list_ = utils.make_point_list_(point_list)
# path = utils.optimized_path(coords=list_, start=[upper_left[0], upper_left[1]])
# for i, point in enumerate(path):
# cv2.putText(img,
#             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#             org=point,
#             text=str(i),
#             thickness=5,
#             fontScale=3,
#             color=(255, 0, 0))
#
# plt.imshow(img)
#
#
# rect = cv2.minAreaRect(point_list)
#
# # pad the rectangle
# (center, (w, h), angle) = rect
# w = w + 10000
# h = h + 1000
# rect = (center, (w, h), angle)
#
# img = Image.open(f'{path_images}/20230525_172420_ESWW0070020_1.JPG')
# img = np.array(img)
# box = cv2.boxPoints(rect) # cv2.cv.BoxPoints(rect) for OpenCV <3.x
# box = np.int0(box)
# cv2.drawContours(img, [box], 0, (255, 0, 0), 5)
# plt.imshow(img)
#
#
# def crop_minAreaRect(img, rect):
#
# # rotate img
# angle = rect[2]
# rows,cols = img.shape[0], img.shape[1]
# M = cv2.getRotationMatrix2D((rows/2,cols/2),angle,1)
# img_rot = cv2.warpAffine(img,M,(cols,rows))
#
# # rotate bounding box
# rect0 = (rect[0], rect[1], 0.0)
# box = cv2.boxPoints(rect0)
# pts = np.int0(cv2.transform(np.array([box]), M))[0]
# pts[pts < 0] = 0
#
# # crop
# img_crop = img_rot[pts[1][1]:pts[0][1],
#                    pts[1][0]:pts[2][0]]
#
# return img_crop
#
# cropped = crop_minAreaRect(img, rect)
#
# plt.imshow(cropped)
#
#
#
