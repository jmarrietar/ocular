#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import random
import shutil 
from os.path import exists
from os import makedirs
from PIL import Image


CLASSES = [0, 1]
SAMPLE = 10000
SOURCE_DIR = "data/voets/train"
random.seed(42)

for CLASS in CLASSES:

    list_images = os.listdir(os.path.join(SOURCE_DIR, str(CLASS)))
    sampled_list = random.sample(list_images, int(SAMPLE/2)) 
    
    # Create directories
    if not exists("sample@{}/train/{}".format(SAMPLE, CLASS)):
        makedirs("sample@{}/train/{}".format(SAMPLE, CLASS))
    
    for image_name in sampled_list: 
        print(image_name)
        
        image_source_path = os.path.join(SOURCE_DIR, str(CLASS), image_name)
        
        if image_source_path.endswith('.jpg'): # Already preprocessed image
            try:
                img = Image.open(image_source_path) 
                img.verify()
            except (IOError, SyntaxError) as e:
                print(e)
                print('Bad file:', image_name)
                
        destination = "sample@{}/train/{}/{}".format(SAMPLE, CLASS, image_name)
        dest = shutil.copyfile(image_source_path, destination)