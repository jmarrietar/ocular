#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
from os import makedirs
from os.path import join, exists
import random
from tfr_image import TFRimage


LABELED = 3000
#random.seed(42)
#random.seed(70)


"""
Importante: 
-----------

TO DO: 
    - Checkear Gradability del banco de Imagenes, al momento de hacer los Samplings. 
    https://raw.githubusercontent.com/mikevoets/jama16-retina-replication/master/vendor/eyepacs/eyepacs_gradability_grades.csv

    - Que los samples de imagenes que estoy testeando sean Gradables. 

TO DO: 
    - Recrear los splits de los datasets pero solo con funciones gradables. 

"""


DIR = "/Volumes/APOLLOM110/server/jama16-retina-replication-master/data/eyepacs/train"
DEST_DIR = "../data/sample@{}".format(LABELED)


# List all training data (for both clases)
images_class_0 = os.listdir(os.path.join(DIR, "0"))
images_class_0 = [os.path.join(DIR, str("0"), x) for x in images_class_0]
images_class_1 = os.listdir(os.path.join(DIR, "1"))
images_class_1 = [os.path.join(DIR, str("1"), x) for x in images_class_1]

"""
Idea: 
    - Funcion Aqui que me lea las imagenes y elimine del listado las que no sean gradables 
    - La funcion lee el archivo
"""

# Get Labeled samples
train_0_sample = random.sample(images_class_0, k=int(LABELED / 2))
train_1_sample = random.sample(images_class_1, k=int(LABELED / 2))

data_partitions = {
    "train": {"0": train_0_sample, "1": train_1_sample},
}

# Create Directories Train/Test and distribute images {0,1} classes
for partition_split, images in data_partitions.items():
    print("Creating directory partition: {}".format(partition_split))

    # Create Directories
    [
        makedirs(join(DEST_DIR, partition_split, str(i)))
        for i in [0, 1]
        if not exists(join(DEST_DIR, partition_split, str(i)))
    ]

    images_0 = images.get("0")
    images_1 = images.get("1")

    print("Moving Images ...")
    for image_source in images_0:
        destination = os.path.join(DEST_DIR, partition_split, "0")
        dest = shutil.copy(image_source, destination)
        


    for image_source in images_1:
        destination = os.path.join(DEST_DIR, partition_split, "1")
        dest = shutil.copy(image_source, destination)

tool = TFRimage()
tool.create_tfrecords(
    dataset_dir=DEST_DIR + "/train", 
    tfrecord_filename="ocular-labeled@{}".format(LABELED),
    num_shards=2,
)
