# ======================================================================================================================
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Project: MixedInf
# Date: 27.10.2021
# ======================================================================================================================

import matplotlib as mpl
import cv2

mpl.use('Qt5Agg')
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import numpy as np


class ImageSelector:

    def __init__(self, dir_images, dir_overlays):
        self.dir_images = dir_images
        self.dir_overlays = dir_overlays

    def get_files_to_process(self):

        # get current date time
        now = datetime.now()
        dt_string = now.strftime("%d%m%Y_%H%M")

        # get all images and their paths
        walker = iter(os.walk(self.dir_images))
        root, dirs, files = next(walker)
        all_images = [Path(root + "/" + file) for file in files]

        # get all overlays and their paths
        walker = iter(os.walk(self.dir_images))
        root, dirs, files = next(walker)
        all_overlays = [Path(root + "/" + file) for file in files]

        # determine which files have already been edited
        path_existing_logfile = Path(f'{self.dir_images}/logfile.csv')
        if path_existing_logfile.exists():
            logfile = pd.read_csv(path_existing_logfile)
            processed_files = logfile['path'].tolist()
            processed = []
            for path in processed_files:
                processed.append(Path(path))
            # save copy of old file version as backup
            logfile.to_csv(f'Z:/Public/Jonas/004_Divers/logfile{dt_string}.csv', index=False)
        # if non have been edited, initiate an empty logfile
        else:
            logfile = pd.DataFrame(columns=['path', 'action'])

        # List of images to edit excluding those already edited
        if 'processed' in locals():
            images_proc = set(all_images) - set(processed)
        else:
            images_proc = all_images
        images_proc = [file for file in images_proc]

        return images_proc, all_overlays, logfile

    @staticmethod
    def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):

        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(image, dim, interpolation=inter)

    def select_images(self):

        images, overlays, logfile = self.get_files_to_process()

        print("found ", len(images), " files to process.")

        log = []
        for path in images:
            img_basename = os.path.basename(path)
            overlay_ = cv2.imread(f'{self.dir_overlays}/{img_basename.replace(".jpg", ".tiff")}')
            resize = self.resize_with_aspect_ratio(overlay_, height=1300)  # Resize by width

            cv2.imshow('ImageWindow', resize)
            key = cv2.waitKey(0)
            if key == ord('k'):  # keep
                action = "None"
                cv2.destroyAllWindows()
            elif key == ord('r'):  # remove
                action = "Exclude"
                cv2.destroyAllWindows()
            elif key == ord('q'):  # quit
                cv2.destroyAllWindows()
                break
            else:
                print('not an interpretable input')

            log.append({'path': path, 'action': action})

        logfile_ = pd.DataFrame(log)
        logfile = logfile.append(logfile_, ignore_index=True)

        logfile.to_csv(f'{self.dir_images}/logfile.csv', index=False)

        return logfile


def run():
    dir_images = "P:/Public/Jonas/004_Divers/PycnidiaPattern_all/processed/overlay"
    dir_overlays = "P:/Public/Jonas/004_Divers/PycnidiaPattern_all/processed/overlay"
    image_selector = ImageSelector(dir_images, dir_overlays)
    image_selector.select_images()


if __name__ == '__main__':
    run()

