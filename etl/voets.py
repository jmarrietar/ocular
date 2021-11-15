#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TO DO: 
    - Remove ungradable images "--only_gradable" 
"""

import csv
import sys
import os
import random
import shutil
import yaml
import logging
import logging.config

from shutil import rmtree
from glob import glob
from os import makedirs, rename
from os.path import join, exists
from tfr_image import TFRimage
from utils import resize_and_center_fundus
from utils.decorators import log_entry_exit, timed

# Files & Directories
#RAW = "../kaggle-raw"
RAW = '/Volumes/APOLLOM110/jama16-retina-replication/data/eyepacs/pool/raw'
DATA_DIR = "data/voets"
TMP_PATH = join(DATA_DIR, "tmp")
TRAIN_LABELS = "data/trainLabels.csv"
TEST_LABELS = "data/testLabels.csv"
#DIAMETER = 299
DIAMETER = 512
RANDOM_SEED = 42
NUM_SHARDS = 8
VALIDATION_SIZE = 0.2

# Logger
with open("log_config.yaml", "r") as f:
    log_cfg = yaml.safe_load(f.read())
logging.config.dictConfig(log_cfg)
logger = logging.getLogger(__name__)


def preprocess_images() -> None:
    """Preprocess Raw images and distribute them to directories
    
    Using labels (train_labels, test_labels) take images from 
    raw directory center, resize, and finally distribute them 
    to directories 0, 1, 2, 3, 4

    """

    data_dir_dict = {"train": TRAIN_LABELS, "test": TEST_LABELS}
    failed_images = []
    img_not_found = []

    logger.info("Creating directories for grades ...")

    [  # Create directories for grades.
        makedirs(join(DATA_DIR, str(i)))
        for i in [0, 1, 2, 3, 4]
        if not exists(join(DATA_DIR, str(i)))
    ]

    # tmp directory for saving temporary preprocessing files.
    if exists(TMP_PATH):
        rmtree(TMP_PATH)
    makedirs(TMP_PATH)

    for key, value in data_dir_dict.items():
        split = key
        labels = value
        with open(labels, "r") as f:
            reader = csv.reader(f, delimiter=",")
            next(reader)

            for i, row in enumerate(reader):
                basename, grade = row[:2]

                try:
                    im_path = glob(join(RAW, split, "{}*".format(basename)))[0]

                    # Find contour of eye fundus in image, and scale
                    #  diameter of fundus to 299 pixels and crop the edges.
                    res = resize_and_center_fundus(
                        save_path=TMP_PATH,
                        image_path=im_path,
                        diameter=DIAMETER,
                        verbosity=0,
                    )

                    # Status message.
                    msg = "\r- Preprocessing image: {0:>7}".format(i + 1)
                    sys.stdout.write(msg)
                    sys.stdout.flush()

                    if res != 1:
                        failed_images.append(basename)
                        continue

                    new_filename = "{0}.jpg".format(basename)

                    # Move the file from the tmp folder to the right grade folder.
                    rename(
                        join(TMP_PATH, new_filename),
                        join(DATA_DIR, str(int(grade)), new_filename),
                    )
                except:
                    img_not_found.append(basename)

    # Clean TMP Folder
    rmtree(TMP_PATH)

    logger.info("\nNot found {} images.".format(len(img_not_found)))
    logger.info("Failed {} images.".format(len(failed_images)))
    logger.info(", ".join(failed_images))


def distribute_images() -> None:
    """Distribute images into Train and Test directories 

    Using directories 0, 1, 2, 3, 4 distribute images 
    into Class 0 [0,1] and Class 1 [2,3,4] for Train and Test partitions.

    Example:

    ├── data
    │   │── train
    │       │── 0
    │       │── 1
    │   │── test
    │       │── 0
    │       │── 1
    
    """

    images_class_0 = []
    images_class_1 = []
    random.seed(RANDOM_SEED)

    # Create lists of images for class 0 and 1
    for class_ in range(0, 2):
        images_list = os.listdir(os.path.join(DATA_DIR, str(class_)))
        images_list = [os.path.join(DATA_DIR, str(class_), x) for x in images_list]
        images_class_0.extend(images_list)

    for class_ in range(2, 5):
        images_list = os.listdir(os.path.join(DATA_DIR, str(class_)))
        images_list = [os.path.join(DATA_DIR, str(class_), x) for x in images_list]
        images_class_1.extend(images_list)

    # Shuffle lists of images
    random.shuffle(images_class_0)
    random.shuffle(images_class_1)

    # Define Number of training images Class 0
    BIN2_0_CNT = 48784
    BIN2_0_TR_CNT = 40688
    TAIL_0_TST = BIN2_0_CNT - BIN2_0_TR_CNT

    # Define Number of Testing images Class 1
    BIN2_1 = len(images_class_1)
    BIN2_1_TR_CNT = 16458
    TAIL_1_TST = BIN2_1 - BIN2_1_TR_CNT

    # Split Train and Test datasets
    training_0 = images_class_0[:BIN2_0_TR_CNT]
    training_1 = images_class_1[:BIN2_1_TR_CNT]

    test_0 = images_class_0[-TAIL_0_TST:]
    test_1 = images_class_1[-TAIL_1_TST:]

    data_partitions = {
        "train": {"0": training_0, "1": training_1},
        "test": {"0": test_0, "1": test_1},
    }

    # Create Directories Train/Test and distribute images {0,1} classes
    for partition_split, images in data_partitions.items():
        logger.info("Creating directory partition: {}".format(partition_split))

        # Create Directories
        [
            makedirs(join(DATA_DIR, partition_split, str(i)))
            for i in [0, 1]
            if not exists(join(DATA_DIR, partition_split, str(i)))
        ]

        images_0 = images.get("0")
        images_1 = images.get("1")

        logger.info("Moving Images ...")
        for image_source in images_0:
            destination = os.path.join(DATA_DIR, partition_split, "0")
            dest = shutil.copy(image_source, destination)

        for image_source in images_1:
            destination = os.path.join(DATA_DIR, partition_split, "1")
            dest = shutil.copy(image_source, destination)


def create_tfrecords_files() -> None:
    """Create TFRecords files for Train and Test partitions"""

    logger.info("Creating TFRecords ...")
    tool = TFRimage()

    # Train TFRecords
    tool.create_tfrecords(
        dataset_dir="data/voets/train",
        tfrecord_filename="voets",
        validation_size=VALIDATION_SIZE,
        num_shards=NUM_SHARDS,
    )

    # Test TFRecords
    tool.create_tfrecords(
        dataset_dir="data/voets/test", tfrecord_filename="voets", num_shards=4,
    )


def clean() -> None:
    """Delete all files except for the TFrecords"""

    # Clean Directory Keep only TFRecords
    [rmtree(join(DATA_DIR, str(i))) for i in [0, 1, 2, 3, 4]]
    [rmtree(join(DATA_DIR, "train", str(i))) for i in [0, 1]]
    [rmtree(join(DATA_DIR, "test", str(i))) for i in [0, 1]]


@timed
@log_entry_exit
def main():
    """Pipeline definition

    Preprocess, distribute and create 
    Train and Test partitions with TFRecords

    """
    preprocess_images()
    distribute_images()
    create_tfrecords_files()
    clean()


if __name__ == "__main__":
    main()
