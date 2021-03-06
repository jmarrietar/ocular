#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
from os import makedirs
from os.path import join, exists
import random
from tfr_image import TFRimage
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from numpy import array
from numpy import argmax
from keras.utils import to_categorical
import pandas as pd 
import numpy as np
import os

DIR = "/Users/jmarrietar/Documents/ocular/data/sample@31000/train"


# List all training data (for both clases)
images_class_0 = os.listdir(os.path.join(DIR, "0"))
images_class_0 = [os.path.join(DIR, str("0"), x) for x in images_class_0]
images_class_1 = os.listdir(os.path.join(DIR, "1"))
images_class_1 = [os.path.join(DIR, str("1"), x) for x in images_class_1]

images_paths = images_class_0 + images_class_1
file_names = [x.split('/')[-1] for x in images_paths]
images_labels = [x.split('/')[-2] for x in images_paths]


# Conseguir el class_label 

label = images_class_1[0].split('/')[-2]

# Con el class_label sacar el class_one_hot 
# define example
data = [0, 1]
data = array(data)
print(data)
# one hot encode
encoded = to_categorical(data)
print(encoded)

to_categorical(np.array([0, 1]), dtype=int).tolist()



# define example
data = images_labels
data = array(data, dtype=int)

# one hot encode
encoded = to_categorical(data, dtype=int)
print(encoded)

encoded = encoded.tolist()


df = pd.DataFrame(list(zip(file_names, images_labels, encoded)), 
               columns =['filename', 'class_label', 'class_one_hot']) 


df['filename'] = 'data/processed/sample@31000/train/'+df['class_label']+'/'+df['filename']


df.to_pickle("df_31000.pkl")


###########################
#       5000    FAKE      #
###########################


DIR = "/Users/jmarrietar/Documents/sample@5000/train"


# List all training data (for both clases)
images_class_0 = os.listdir(os.path.join(DIR, "0"))
images_class_0 = [os.path.join(DIR, str("0"), x) for x in images_class_0]
images_class_1 = os.listdir(os.path.join(DIR, "1"))
images_class_1 = [os.path.join(DIR, str("1"), x) for x in images_class_1]

images_paths = images_class_0 + images_class_1
file_names = [x.split('/')[-1] for x in images_paths]
images_labels = [x.split('/')[-2] for x in images_paths]



# Conseguir el class_label 


label = images_class_1[0].split('/')[-2]

# Con el class_label sacar el class_one_hot 
import numpy as np
from numpy import array
from numpy import argmax
from keras.utils import to_categorical
# define example
data = [0, 1]
data = array(data)
print(data)
# one hot encode
encoded = to_categorical(data)
print(encoded)

to_categorical(np.array([0, 1]), dtype=int).tolist()



from numpy import array
from numpy import argmax
from keras.utils import to_categorical
# define example
data = images_labels
data = array(data, dtype=int)

# one hot encode
encoded = to_categorical(data, dtype=int)
print(encoded)

encoded = encoded.tolist()

import pandas as pd 


df = pd.DataFrame(list(zip(file_names, images_labels, encoded)), 
               columns =['filename', 'class_label', 'class_one_hot']) 


df['filename'] = 'data/processed/sample@5000/train/'+df['class_label']+'/'+df['filename']


# FAKE DATA 
df['class_label'] = df['class_label'].iloc[0]
ans =[df['class_one_hot'].iloc[0]] *len(df)
df['class_one_hot'] = ans

###########################################

df.to_pickle("df_fake_5000.pkl")


######################
#     MESSIDOR       #
######################



DIR = "/Users/jmarrietar/Documents/messidor2"



# List all training data (for both clases)
images_class_0 = os.listdir(os.path.join(DIR, "0"))
images_class_0 = [os.path.join(DIR, str("0"), x) for x in images_class_0]
images_class_1 = os.listdir(os.path.join(DIR, "1"))
images_class_1 = [os.path.join(DIR, str("1"), x) for x in images_class_1]

images_paths = images_class_0 + images_class_1
file_names = [x.split('/')[-1] for x in images_paths]
images_labels = [x.split('/')[-2] for x in images_paths]



# Con el class_label sacar el class_one_hot 
import numpy as np
from numpy import array
from numpy import argmax
from keras.utils import to_categorical
# define example
data = [0, 1]
data = array(data)
print(data)
# one hot encode
encoded = to_categorical(data)
print(encoded)

to_categorical(np.array([0, 1]), dtype=int).tolist()



from numpy import array
from numpy import argmax
from keras.utils import to_categorical
# define example
data = images_labels
data = array(data, dtype=int)

# one hot encode
encoded = to_categorical(data, dtype=int)
print(encoded)

encoded = encoded.tolist()

import pandas as pd 


df = pd.DataFrame(list(zip(file_names, images_labels, encoded)), 
               columns =['filename', 'class_label', 'class_one_hot']) 


df['filename'] = 'data/processed/messidor2/'+df['class_label']+'/'+df['filename']


#df2 = df.sample(n=200, random_state=1)

df.to_pickle("df_messidor.pkl")

########################
#     VOETS TEST       #
########################

DIR = "/Users/jmarrietar/Documents/test"


# List all training data (for both clases)
images_class_0 = os.listdir(os.path.join(DIR, "0"))
images_class_0 = [os.path.join(DIR, str("0"), x) for x in images_class_0]
images_class_1 = os.listdir(os.path.join(DIR, "1"))
images_class_1 = [os.path.join(DIR, str("1"), x) for x in images_class_1]

images_paths = images_class_0 + images_class_1
file_names = [x.split('/')[-1] for x in images_paths]
images_labels = [x.split('/')[-2] for x in images_paths]



# Con el class_label sacar el class_one_hot 

# define example
data = [0, 1]
data = array(data)
print(data)
# one hot encode
encoded = to_categorical(data)
print(encoded)

to_categorical(np.array([0, 1]), dtype=int).tolist()




# define example
data = images_labels
data = array(data, dtype=int)

# one hot encode
encoded = to_categorical(data, dtype=int)
print(encoded)

encoded = encoded.tolist()



df = pd.DataFrame(list(zip(file_names, images_labels, encoded)), 
               columns =['filename', 'class_label', 'class_one_hot']) 


df['filename'] = 'data/processed/test/'+df['class_label']+'/'+df['filename']


#df2 = df.sample(n=200, random_state=1)

df.to_pickle("df_voets_test.pkl")







