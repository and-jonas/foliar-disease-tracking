
from RoiAligner import RoiAligner

workdir = 'Z:/Public/Jonas/Data/ESWW007/SingleLeaf/'


def run():
    path_labels = "Z:/Public/Jonas/Data/ESWW007/SingleLeaf/*/JPEG_cam/runs/pose/predict*/labels"
    path_images = "Z:/Public/Jonas/Data/ESWW007/SingleLeaf/*/JPEG_cam"
    path_output = f'{workdir}/Output'
    roi_aligner = RoiAligner(
        path_labels=path_labels,
        path_images=path_images,
        path_output=path_output,
        n_cpus=4,
    )
    roi_aligner.process_all()


if __name__ == "__main__":
    run()