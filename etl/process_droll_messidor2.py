import os
import pandas as pd
from shutil import rmtree

DIR = "data/droll-messidor2/complete/"

# Replace image format
images_files = os.listdir(DIR)

for image_name in images_files:
    new_name = image_name.replace(".tif", ".jpg")
    os.rename(DIR+image_name, DIR+new_name)
    
messidor2_labels_lesions = pd.read_csv("data/droll-messidor2/messidor2.csv")

# Drop NA rows
messidor2_labels_lesions.dropna(inplace=True)

# Replace ."tif" for ".jpg"
messidor2_labels_lesions['Nombre de la imagen'] = messidor2_labels_lesions['Nombre de la imagen'].str.replace(".tif", ".jpg")

# Save lesion labels 
messidor2_labels_lesions.to_csv("data/droll-messidor2/messidor2_lesions.csv", index=False)


# Remove Images that we dont have lesion information
images_files = os.listdir(DIR)
messidor2_labels_lesions = pd.read_csv("data/droll-messidor2/messidor2_lesions.csv")

droll_images = messidor2_labels_lesions['Nombre de la imagen'].to_list()

for image in images_files:
    if not image in droll_images:
        print("Delete image {}".format(image))
        os.remove(DIR+image)