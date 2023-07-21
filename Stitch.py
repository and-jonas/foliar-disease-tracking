# import the necessary packages
import utils
from Test import Stitcher
import imutils
import cv2
import numpy as np
import scipy
import glob
import os
from pathlib import Path
import imageio
from PIL import Image

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


# image_list = glob.glob("Z:/Public/Jonas/Data/ESWW007/TestData/SingleLeaf/JPEG/*.JPG")
all_images = glob.glob("Z:/Public/Jonas/Data/ESWW007/SingleLeaf/*/JPEG_cam/*.JPG")
plots = ["_".join(os.path.basename(f).replace(".JPG", "").split("_")[2:4]) for f in all_images]
leaf_list = np.unique(np.array(plots)).tolist()

series = []
for leaf in leaf_list:
    series.append([i for i in all_images if f'{leaf}.JPG' in i])

s = series[0]

for image in s:
    ref_img = Image.open(image)
    ref_img = np.array(ref_img)

    im_blur = cv2.medianBlur(ref_img, 15)
    im_blur_hsv = cv2.cvtColor(im_blur, cv2.COLOR_RGB2HSV)

    lower = np.array([95, 120, 22])  # v changed successively from 35 to 30 to 22 for selected images
    upper = np.array([115, 220, 255])
    mask = cv2.inRange(im_blur_hsv, lower, upper)
    plt.imshow(ref_img)

    mask_ = scipy.ndimage.binary_fill_holes(mask)
    leaf_mask = np.bitwise_not(mask_)
    leaf_mask = np.where(leaf_mask, 255, 0).astype("uint8")
    _, output, stats, ctr = cv2.connectedComponentsWithStats(leaf_mask, connectivity=4)
    sizes = stats[1:, -1]
    idx = np.argmax(sizes)
    leaf_mask = np.in1d(output, idx+1).reshape(output.shape)
    leaf_mask = np.where(leaf_mask, 255, 0).astype("uint8")


image_list = image_list[10:18]
image_names = [os.path.basename(i).replace(".JPG", "") for i in image_list]

path = "Z:/Public/Jonas/Data/ESWW007/TestData/SingleLeaf"

Hs = []
for i, img_name in enumerate(image_names[:7]):
    print(i)
    img_path = f'{path}/JPEG/{image_names[i]}.JPG'
    leaf_mask_path = f'{path}/output/leaf_mask/{image_names[i]}.png'
    imageA_ = cv2.imread(img_path)
    imageA_ = cv2.cvtColor(imageA_, cv2.COLOR_BGR2RGB)
    maskA = cv2.imread(leaf_mask_path)
    maskA = cv2.cvtColor(maskA, cv2.COLOR_BGR2GRAY)
    maskA = np.uint8(maskA)
    img_path = f'{path}/JPEG/{image_names[i+1]}.JPG'
    leaf_mask_path = f'{path}/output/leaf_mask/{image_names[i+1]}.png'
    imageB_ = cv2.imread(img_path)
    imageB_ = cv2.cvtColor(imageB_, cv2.COLOR_BGR2RGB)
    maskB = cv2.imread(leaf_mask_path)
    maskB = cv2.cvtColor(maskB, cv2.COLOR_BGR2GRAY)
    maskB = np.uint8(maskB)

    # print("done")
    imageA = imutils.resize(imageA_, height=500)
    imageB = imutils.resize(imageB_, height=500)
    maskA = imutils.resize(maskA, height=500)
    maskB = imutils.resize(maskB, height=500)

    # fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    # axs[0].imshow(imageA)
    # axs[0].set_title('original')
    # axs[1].imshow(maskA)
    # axs[1].set_title('transformed')
    # plt.show(block=True)

    # stitch the images together to create a panorama
    stitcher = Stitcher()
    (result, vis, H) = stitcher.stitch(images=[imageA, imageB],
                                       masks=[maskA, maskB],
                                       # masks=[None, None],
                                       showMatches=True)

    imageio.imwrite(f'{path}/output/matches/{image_names[i]}.JPG', vis)
    imageio.imwrite(f'{path}/output/result/{image_names[i]}.JPG', result)

    Hs.append(H)


img_path = f'{path}/JPEG/{image_names[0]}.JPG'
source = cv2.imread(img_path)
source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
source = imutils.resize(source, height=500)

img_path = f'{path}/JPEG/{image_names[4]}.JPG'
target = cv2.imread(img_path)
target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
target = imutils.resize(target, height=500)

# dim_small = np.float32([[0, 0], [imageA.shape[1], 0], [imageA.shape[1], imageA.shape[0]], [0, imageA.shape[0]]])
# dim_large = np.float32([[0, 0], [imageB_.shape[1], 0], [imageB_.shape[1], imageB_.shape[0]], [0, imageB_.shape[0]]])

dim_small = np.float32([[0, 0], [749, 0], [749, 500], [0, 500]])
dim_large = np.float32([[0, 0], [8192, 0], [8192, 5464], [0, 5464]])

H_scale = cv2.getPerspectiveTransform(src=dim_small,
                                      dst=dim_large)

H1 = Hs[0]
H2 = Hs[1]
H3 = Hs[2]
H4 = Hs[3]
H5 = Hs[4]

H_0 = np.dot(H1, H2)
H = np.dot(H_0, H3)
Hx = np.dot(H, H4)

result = cv2.warpPerspective(source, Hx,
                             (source.shape[1] + target.shape[1], source.shape[0]))


fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
axs[0].imshow(source)
axs[0].set_title('source')
axs[1].imshow(target)
axs[1].set_title('target')
axs[2].imshow(result)
axs[2].set_title('trsfrmd')
plt.show(block=True)

new = np.dot(Hs[0], Hs[1])
out = cv2.warpPerspective(imageA_, new,
                          (imageA_.shape[1] + imageB_.shape[1], imageA_.shape[0]))







result = cv2.warpPerspective(imageA_, H,
                             (imageA_.shape[1] + imageB_.shape[1], imageA_.shape[0]))

plt.imshow(result)





# show the images
cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
# cv2.waitKey(0)


# load the two images and resize them to have a width of 400 pixels
# (for faster processing)
imageA = cv2.imread("Z:/Public/Jonas/Data/ESWW007/TestData/SingleLeaf/JPEG/BF0A2336.JPG")
imageB = cv2.imread("Z:/Public/Jonas/Data/ESWW007/TestData/SingleLeaf/JPEG/BF0A2337.JPG")
imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB)
imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2RGB)

maskA = cv2.imread("Z:/Public/Jonas/Data/ESWW007/TestData/SingleLeaf/output/leaf_mask/BF0A2336.png")
maskA = cv2.cvtColor(maskA, cv2.COLOR_BGR2GRAY)
maskA = np.uint8(maskA)
maskB = cv2.imread("Z:/Public/Jonas/Data/ESWW007/TestData/SingleLeaf/output/leaf_mask/BF0A2337.png")
maskB = cv2.cvtColor(maskB, cv2.COLOR_BGR2GRAY)
maskB = np.uint8(maskB)

# imageA = imageA[1500:3000, 0:5000]
# imageB = imageB[1800:3300, ]

# leaf_mask = np.zeros_like(imageA[:, :, 0])
# leaf_mask = np.where(imageA[:, :, 2] >= 180, 0, 1).astype("uint8")
# leaf_mask = cv2.erode(leaf_mask, np.ones((15, 15), np.uint8))
# leaf_mask = cv2.dilate(leaf_mask, np.ones((19, 19), np.uint8))
# leaf_mask_A = scipy.ndimage.morphology.binary_fill_holes(leaf_mask).astype("uint8")
#
# plt.imshow(leaf_mask)


# plt.imshow(imA)

print("done")
imageA = imutils.resize(imageA, height=750)
imageB = imutils.resize(imageB, height=750)
maskA = imutils.resize(maskA, height=750)
maskB = imutils.resize(maskB, height=750)
# stitch the images together to create a panorama
stitcher = Stitcher()
(result, vis, H) = stitcher.stitch(images=[imageA, imageB],
                                   masks=[maskA, maskB],
                                   # masks=[None, None],
                                   showMatches=True)

dim_small = np.float32([[0, 0], [749, 0], [749, 500], [0, 500]])
dim_large = np.float32([[0, 0], [8192, 0], [8192, 5464], [0, 5464]])

H_scale = cv2.getPerspectiveTransform(src=dim_small,
                                      dst=dim_large)

