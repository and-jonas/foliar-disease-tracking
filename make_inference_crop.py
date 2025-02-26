import os
import utils

os.chdir("Z:/Public/Jonas/Data/ESWW009/SingleLeaf")

pattern_imgs = "202*/JPEG_cam"
pattern_coords = "202*/JPEG_cam/runs/pose/predict/labels"
output_dir = "inference_crops"

# make crops that maintain image width but sample only part of image in height direction
utils.make_inference_crops(pattern_imgs=pattern_imgs,
                           pattern_coords=pattern_coords,
                           output_dir=output_dir,
                           img_sz=(5464, 8192),
                           patch_sz=(2048, 8192))

