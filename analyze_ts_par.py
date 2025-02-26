
from TSAnalyzer import TSAnalyzer
import numpy as np
# import os

# workdir = 'Z:/Public/Jonas/Data/ESWW009/SingleLeaf/Output'
# workdir = "/home/anjonas/public/Public/Jonas/Data/ESWW007/SingleLeaf/Output"
workdir = "Z:/Public/Jonas/Data/ESWW007/SingleLeaf/Output"


def run():
    path_images = f'{workdir}/*/result/piecewise'
    path_aligned_masks = f'{workdir}/*/mask_aligned/piecewise'
    path_kpts = f'{workdir}/*/keypoints'
    # path_output = "/home/anjonas/public/Public/Jonas/011_STB_leaf_tracking/output/ts_test"
    path_output = "Z:/Public/Jonas/011_STB_leaf_tracking/output/ts_test"
    ts_analyzer = TSAnalyzer(
        path_aligned_masks=path_aligned_masks,
        path_images=path_images,
        path_kpts=path_kpts,
        path_output=path_output,
        n_cpus=1,
    )
    ts_analyzer.process_all()


if __name__ == "__main__":
    run()

