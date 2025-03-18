
from TSAnalyzer import TSAnalyzer
import argparse

# parse arguments
parser = argparse.ArgumentParser(description="Run tracking.")
parser.add_argument("--experiment", type=str, required=True, help="Experiment ID")
parser.add_argument("--start", type=int, required=True, help="Series indices to process")
parser.add_argument("--end", type=int, required=True, help="Series indices to process")
parser.add_argument("--n_cpus", type=int, required=True, help="")
args = parser.parse_args()
experiment = args.experiment
start = args.start
end = args.end
n_cpus = args.n_cpus

# experiment = "ESWW007"
# start = 0
# end = 1
# n_cpus = 1

# variables
workdir = f"/home/anjonas/public/Public/Jonas/Data/{experiment}/SingleLeaf/Output"
path_out = f"/home/anjonas/public/Public/Jonas/Data/{experiment}/SingleLeaf/Output_ts"


def run():
    path_images = f'{workdir}/*/result/piecewise'
    path_aligned_masks = f'{workdir}/*/mask_aligned/piecewise'
    path_kpts = f'{workdir}/*/keypoints'
    ts_analyzer = TSAnalyzer(
        path_aligned_masks=path_aligned_masks,
        path_images=path_images,
        path_kpts=path_kpts,
        path_output=path_out,
        n_cpus=n_cpus,
        indices=(start, end)
    )
    ts_analyzer.process_all()


if __name__ == "__main__":
    run()

