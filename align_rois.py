
# from RoiAligner import RoiAligner
from RoiAligner2 import RoiAligner

import os
os.getcwd()
# workdir = '/home/anjonas/public/Public/Jonas/Data/ESWW007/SingleLeaf'
# workdir = 'Z:/Public/Jonas/Data/ESWW009/SingleLeaf'
workdir = "/home/anjonas/public/Public/Jonas/Data/ESWW009/SingleLeaf"
# modeldir = 'Z:/Public/Jonas/Data/ESWW006/Images_trainset/Output/Models/rf_segmentation_v3.pkl'
modeldir = '/home/anjonas/public/Public/Jonas/Data/ESWW006/Images_trainset/Output/Models/rf_segmentation_v3.pkl'


def run():
    path_labels = f'{workdir}/*/JPEG_cam/runs/pose/predict/labels'
    path_images = f'{workdir}/*/JPEG_cam'
    path_leaf_masks = '/home/anjonas/public/Public/Jonas/011_STB_leaf_tracking/predictions'
    path_output = f'{workdir}/Output'
    path_model = modeldir
    # path_sample_list = f'{workdir}/Output/unmatched_all_orig.txt'
    roi_aligner = RoiAligner(
        path_labels=path_labels,
        path_images=path_images,
        path_leaf_masks=path_leaf_masks,
        path_output=path_output,
        path_model=path_model,
        # path_sample_list=path_sample_list,
        n_cpus=1,
    )
    roi_aligner.process_all()


if __name__ == "__main__":
    run()


