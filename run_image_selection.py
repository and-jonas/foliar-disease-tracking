# ======================================================================================================================
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Project: MixedInf
# Date: 27.10.2021
# Sample images amenable to further analysis
# ======================================================================================================================

from image_selector import ImageSelector


def run():
    dir_images = "P:/Public/Jonas/004_Divers/PycnidiaPattern_all/processed/overlay"
    dir_overlays = "P:/Public/Jonas/004_Divers/PycnidiaPattern_all/processed/overlay"
    image_selector = ImageSelector(dir_images, dir_overlays)
    image_selector.select_images()


if __name__ == '__main__':
    run()