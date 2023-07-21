import glob
import os
from pathlib import Path
from importlib import reload

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import cv2
import imageio

from skimage import transform
import utils

from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import color, exposure
from skimage.morphology import extrema
from skimage.measure import label
import scipy

reload(utils)

path = Path("Z:/Public/Jonas/Data/ESWW007/TestData/SingleLeaf")

out_dir_checker = path / "output" / "checker"
out_dir_mask = path / "output" / "mask"
out_dir_leaf_mask = path / "output" / "leaf_mask"

out_dir_checker.mkdir(parents=True, exist_ok=True)
out_dir_mask.mkdir(parents=True, exist_ok=True)
out_dir_leaf_mask.mkdir(parents=True, exist_ok=True)

files = glob.glob(f"{path}/JPEG/*.JPG")

ctr_coord = []
for f in files[10:18]:

    print(f)

    base_name = os.path.basename(f)
    png_name = base_name.replace(".JPG", ".png")

    ref_img = Image.open(f)
    ref_img = np.array(ref_img)

    im_blur = cv2.medianBlur(ref_img, 15)
    im_blur_hsv = cv2.cvtColor(im_blur, cv2.COLOR_RGB2HSV)

    lower = np.array([95, 120, 22])  # v changed successively from 35 to 30 to 22 for selected images
    upper = np.array([115, 220, 255])
    mask = cv2.inRange(im_blur_hsv, lower, upper)
    # plt.imshow(mask)

    mask_ = scipy.ndimage.binary_fill_holes(mask)
    leaf_mask = np.bitwise_not(mask_)
    leaf_mask = np.where(leaf_mask, 255, 0).astype("uint8")
    _, output, stats, ctr = cv2.connectedComponentsWithStats(leaf_mask, connectivity=4)
    sizes = stats[1:, -1]
    idx = np.argmax(sizes)
    leaf_mask = np.in1d(output, idx+1).reshape(output.shape)
    leaf_mask = np.where(leaf_mask, 255, 0).astype("uint8")
    # plt.imshow(out)

    im = cv2.cvtColor(im_blur, cv2.COLOR_RGB2GRAY)

    # fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    # axs[0].imshow(im)
    # axs[0].set_title('original')
    # axs[1].imshow(im_eq)
    # axs[1].set_title('transformed')
    # plt.show(block=True)

    im = cv2.normalize(im, 0, 255, norm_type=cv2.NORM_MINMAX)
    im = np.where(mask_, 95, im)
    image = exposure.rescale_intensity(im, in_range=(50, 180))

    # Comparison between image_max and im to find the coordinates of local maxima
    coordinates = peak_local_max(image,
                                 threshold_abs=130,
                                 threshold_rel=0.8,
                                 footprint=np.ones((151, 151)),
                                 min_distance=150,
                                 exclude_border=False)

    # # display results
    # fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)
    # ax = axes.ravel()
    # ax[0].imshow(im, cmap=plt.cm.gray)
    # ax[0].axis('off')
    # ax[0].set_title('Original')
    #
    # ax[1].imshow(im, cmap=plt.cm.gray)
    # ax[1].autoscale(False)
    # ax[1].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    # ax[1].axis('off')
    # ax[1].set_title('Peak local max')
    #
    # fig.tight_layout()
    #
    # plt.show()

    checker_ref, ctr_ref, image_ref = utils.find_marker_centroids(image=ref_img, size_th=150, coordinates=coordinates,
                                                                  leaf_mask=leaf_mask)
    checker_ref = checker_ref.astype("uint8")

    out_name_checker = out_dir_checker / base_name
    out_name_mask = out_dir_mask / base_name
    out_name_leaf_mask = out_dir_leaf_mask / png_name

    imageio.imwrite(out_name_checker, image_ref)
    imageio.imwrite(out_name_mask, checker_ref)
    imageio.imwrite(out_name_leaf_mask, leaf_mask)

    ctr_coord.append(ctr_ref)

c = utils.flatten_centroid_data(input=ctr_coord[0], asarray=True)
tl, tr = utils.find_top_left(pts=c)

pts = np.float32(c).reshape(-1, 1, 2)
cv2.perspectiveTransform(pts, H)



plt.imshow(image_ref)


fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
axs[0].imshow(ref_img)
axs[0].set_title('original')
axs[1].imshow(checker_ref)
axs[1].set_title('transformed')
plt.show(block=True)

ref_img = Image.open(files[0])
ref_img = np.array(ref_img)

im_blur = cv2.medianBlur(ref_img, 15)
im = color.rgb2gray(im_blur)


plt.imshow(im)

test = cv2.medianBlur(ref_img, 51)
test_hsv = color.rgb2hsv(ref_img)

WTF = test_hsv.astype(np.uint8)

lower = np.array([95/255, 150/255, 22/255])  # v changed successively from 35 to 30 to 22 for selected images
upper = np.array([115/255, 200/255, 255/255])
mask = cv2.inRange(test_hsv, lower, upper)

plt.imshow(test_hsv)


# image_max is the dilation of im with a 20*20 structuring element
# It is used within peak_local_max function
image_max = ndi.maximum_filter(im, size=50, mode='constant')

# Comparison between image_max and im to find the coordinates of local maxima
coordinates = peak_local_max(im,
                             threshold_abs=0.65,
                             footprint=np.ones((151, 151)),
                             min_distance=100,
                             exclude_border=False)












for f in files[1:]:
    im1 = Image.open(f)
    im1 = np.array(im1)


im1 = Image.open("Z:/Public/Jonas/Data/ESWW007/TestData/SingleLeaf/JPEG/BF0A2332.JPG")
im1 = np.array(im1)
im2 = Image.open("Z:/Public/Jonas/Data/ESWW007/TestData/SingleLeaf/JPEG/BF0A2334.JPG")
im2 = np.array(im2)

# plt.imshow(im1)

checker1, ctr1, image1 = utils.find_marker_centroids(image=im1, size_th=2000, dir="smaller")
checker2, ctr2, image2 = utils.find_marker_centroids(image=im2, size_th=2000, dir="smaller")


tform = transform.estimate_transform('similarity', ctr2, ctr1)

np.allclose(tform.inverse(tform(ctr2)), ctr2)

out = transform.warp(im2, inverse_map=tform.inverse)

plt.imshow(out)

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
axs[0].imshow(im1)
axs[0].set_title('original')
axs[1].imshow(out)
axs[1].set_title('transformed')
plt.show(block=True)

tf = transform.PolynomialTransform()
tf.estimate(ctr2, ctr1)
transformed = transform.PolynomialTransform(im2, tf)




mask = cv2.medianBlur(mask, 13)


im2 = Image.open("Z:/Public/Jonas/Data/ESWW007/TestData/SingleLeaf/JPEG/BF0A2334.JPG")
im2 = np.array(im2)
# plt.imshow(im2)

pts2 = np.float32([[348, 2894], [6894, 3211], [7156, 3807], [299, 3472]])
pts1 = np.float32([[514, 2609], [6976, 2707], [7274, 3226], [482, 3182]])

# pts1 = np.float32([[2604, 502],[2583, 6972],[3091, 7286],[3178, 482]])
# pts2 = np.float32([[2609, 514],[2707, 6976],[3226, 7274],[3182, 482]])

M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(im2,M,(8192, 5464))
plt.subplot(121),plt.imshow(im1),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()

imageio.imwrite("Z:/Public/Jonas/Data/ESWW007/TestData/SingleLeaf/JPEG/test_adj.JPG", dst)