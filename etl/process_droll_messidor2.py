import os
import pandas as pd

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
