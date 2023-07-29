
from RoiAligner import RoiAligner

# workdir = 'Z:/Public/Jonas/Data/ESWW007/SingleLeaf'
workdir = "/home/anjonas/public/Public/Jonas/Data/ESWW007/SingleLeaf"


def run():
    path_labels = f'{workdir}/*/JPEG_cam/runs/pose/predict*/labels'
    path_images = f'{workdir}/*/JPEG_cam'
    path_output = f'{workdir}/Output'
    roi_aligner = RoiAligner(
        path_labels=path_labels,
        path_images=path_images,
        path_output=path_output,
        n_cpus=12,
    )
    roi_aligner.process_all()


if __name__ == "__main__":
    run()
    run()
