#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
os.makedirs('sample@{}'.format(SAMPLE), exist_ok=True)
os.makedirs('sample@{}/train/0'.format(SAMPLE), exist_ok=True)
os.makedirs('sample@{}/train/1'.format(SAMPLE), exist_ok=True)
"""

import os
import random
import shutil 

from os import listdir
from PIL import Image


CLASS = 0
SAMPLE = 20000
SOURCE_DIR = "train_voets"

list_images = os.listdir(os.path.join(SOURCE_DIR, str(CLASS)))
sampled_list = random.sample(list_images, int(SAMPLE/2)) 


for image_name in sampled_list: 
    
    image_source_path = os.path.join(SOURCE_DIR, str(CLASS), image_name)
    
    if image_source_path.endswith('.jpg'):
        try:
            img = Image.open(image_source_path) 
            img.verify()
        except (IOError, SyntaxError) as e:
            print(e)
            print('Bad file:', image_name)
            
    destination = "sample@{}/train/{}/{}".format(SAMPLE, CLASS, image_name)
    dest = shutil.copyfile(image_source_path, destination) 






