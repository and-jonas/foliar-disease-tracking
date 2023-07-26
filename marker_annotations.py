
# ======================================================================================================================
# Transforms key point annotations exported from CVAT in 'CVAT for images 1.1' format
# to the format required for ultralytics YOLOv8
# ======================================================================================================================

import pandas as pd
from bs4 import BeautifulSoup
import glob
import os

# load annotation file
with open('Z:/Public/Jonas/Data/ESWW007/CvatDatasets/Markers/Set1/annotations.xml', 'r') as f:
    data = f.read()

# Passing the stored data inside
# the beautifulsoup parser
Bs_data = BeautifulSoup(data, "xml")

# Finding all instances of tag `image`
b_unique = Bs_data.find_all('image')

# get list of all image names
image_names = glob.glob('Z:/Public/Jonas/Data/ESWW007/CvatDatasets/Markers/Set1/Images/*.JPG')
image_names = [os.path.basename(x) for x in image_names]

# iterate over all annotated images
for image_name in image_names:
    # extract attributes of the first instance of the tag
    b_name = Bs_data.find('image', {'name': image_name})
    value = b_name.find_all('points')
    v = [value[i].get('points') for i in range(len(value))]

    out = []
    for element in v:
        out.extend([x.split(",") for x in element.split(";")])

    result = []
    for p in out:
        print(p)
        cl = 0
        c_x = int(float(p[0]))/8192  # transform into relative coordinates
        c_y = int(float(p[1]))/5464
        s1 = 0.00800  # approximate size of a marker
        s2 = s1
        x = c_x
        y = c_y
        result.append([cl, c_x, c_y, s1, s2, x, y])

    # create output .txt file
    df = pd.DataFrame(data=result)
    txt_name = image_name.replace(".JPG", ".txt")
    df.to_csv(f"Z:/Public/Jonas/Data/ESWW007/CvatDatasets/Markers/Set1/{txt_name}",
              sep=" ",
              index=False,
              header=False)




