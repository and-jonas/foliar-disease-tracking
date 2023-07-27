
# ======================================================================================================================
# Verifies that the extracted image rotation matrix, bounding box coordinates,
# and final transformation matrix are correct.
# ======================================================================================================================

from PIL import Image
import numpy as np
import json
import cv2
import skimage

# get image
p = "Z:/Public/Jonas/Data/ESWW007/SingleLeaf/20230608/JPEG_cam/20230608_163537_ESWW0070020_10.JPG"
img = Image.open(p)
img = np.asarray(img)

# get bounding box localization info
output_p = "Z:/Public/Jonas/Data/ESWW007/SingleLeaf/Output/ESWW0070020_10/roi/20230608_163537_ESWW0070020_10.json"
f = open(output_p)
data = json.load(f)
rot = np.asarray(data['rotation_matrix'])
bbox = np.asarray(data['bounding_box'])
tform = np.asarray(data['transformation_matrix'])
f.close()

# rotate the image
rows, cols = img.shape[0], img.shape[1]
img_rot = cv2.warpAffine(img, rot, (cols, rows))

# add the bounding box
box_ = np.intp(bbox)
cv2.drawContours(img_rot, [box_], 0, (255, 0, 0), 9)

# perform the transformation
roi = img_rot[box_[0][1]:box_[2][1], box_[0][0]:box_[1][0]]
warped = skimage.transform.warp(roi, tform, output_shape=(629, 5129))
warped = skimage.util.img_as_ubyte(warped)
# OK!!
