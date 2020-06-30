#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementation based on code from: 
    https://github.com/mikevoets/jama16-retina-replication
"""


import argparse
import csv
import sys
from shutil import rmtree
import os
from os import makedirs, rename
from os.path import join, exists
from lib.preprocess import resize_and_center_fundus

parser = argparse.ArgumentParser(description="Preprocess EyePACS data set.")
parser.add_argument(
    "--data_dir", help="Directory where EyePACS resides.", default="data/kaggle"
)
parser.add_argument(
    "--raw_dir", help="Directory where raw data resides.", default="data/kaggle"
)
parser.add_argument(
    "--proc_dir", help="Destination for processed data", default="data/kaggle"
)
parser.add_argument("--labels", help="Path for Labels", default="data/kaggle")
parser.add_argument("--img_type", help="Image format", default=None)

args = parser.parse_args()
data_dir = str(args.data_dir)
raw_dir = str(args.raw_dir)
proc_dir = str(args.proc_dir)
data_labels = str(args.labels)
img_type = args.img_type

# Create directories for grades.
[
    makedirs(join(proc_dir, str(i)))
    for i in [0, 1, 2, 3, 4]
    if not exists(join(proc_dir, str(i)))
]

# Create a tmp directory for saving temporary preprocessing files.
tmp_path = join(proc_dir, "tmp")
if exists(tmp_path):
    rmtree(tmp_path)
makedirs(tmp_path)
failed_images = []


# for labels in [train_labels, test_labels]:
for labels in [data_labels]:
    with open(labels, "r") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)

        for i, row in enumerate(reader):
            basename, grade = row[:2]
            try:

                im_path = (
                    join(raw_dir, basename)
                    if img_type is None
                    else join(raw_dir, "{}.{}".format(basename, img_type))
                )

                # Find contour of eye fundus in image, and scale
                #  diameter of fundus to 299 pixels and crop the edges.
                res = resize_and_center_fundus(
                    save_path=tmp_path, image_path=im_path, diameter=299, verbosity=0
                )

                # Status message.
                msg = "\r- Preprocessing image: {0:>7}".format(i + 1)
                sys.stdout.write(msg)
                sys.stdout.flush()

                if res != 1:
                    failed_images.append(basename)
                    continue

                base = os.path.splitext(basename)[0]
                new_filename = "{0}.jpg".format(base)

                # Move the file from the tmp folder to the right grade folder.
                rename(
                    join(tmp_path, new_filename),
                    join(proc_dir, str(int(grade)), new_filename),
                )

            except Exception as e:
                print("Error moving image {} {}".format(basename, e))


# Clean tmp folder.
rmtree(tmp_path)

print("Could not preprocess {} images.".format(len(failed_images)))
print(", ".join(failed_images))
