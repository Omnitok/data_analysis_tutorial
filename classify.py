#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 16:34:11 2023

@author: stejan
ICE CRYSTAL CLASSIFICATION
FOLLOWING: https://www.tensorflow.org/tutorials/images/classification
Returns a matrix (txt file) form with:
    filename
    class #1 (from the model prediction)
    AED
    min_circle
    max_dia
    aspect_ratio
    x
    y
    pressure
    height
    temperature
    RH
    border_flag
"""
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pandas as pd
import matplotlib.image as mpimg
import os
import cv2
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from datetime import datetime, timedelta
from pathlib import Path

def calculate_aed(binary_image_path): # Area Equivalent Diameter
    # Read the binary image
    binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Count white pixels
    white_pixel_count = np.sum(binary_image == 255)
    
    # Calculate area equivalent diameter
    aed = np.sqrt(4 * white_pixel_count / np.pi) * 1.65 # um/px
    
    return aed

# IMPORT DATASET
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, "classification", "particles")
data_dir = pathlib.Path(data_dir)
batch_size = 16
img_width = 180
img_height = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

#%% LOAD THE MODEL
model_path = os.path.join(script_dir, "classification", "classifier_0510.h5")
model = tf.keras.models.load_model(model_path)

#class_names = train_ds.class_names
class_names = ['columns', 'compact', 'irregular', 'plates', 'rosettes']
print(class_names)

#%% PREDICT ON NEW DATA
crystal_path = os.path.join(script_dir, "snow_crystal_segmentation/scs_out/particles/")

particle_mask_path = os.path.join(script_dir, "snow_crystal_segmentation/scs_out/mask_particle/")

save_dir = os.path.join(script_dir, "classification/output/")

save_dir_path = Path(save_dir)

if not save_dir_path.exists():
    save_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Created {save_dir} folder")
else:
    print("Folder exists")

# create the dataframe
data = pd.DataFrame(columns=["Particle_ID", "category", "AED", "min_circle", "max_dia", "AR", "x", "y", "P", "height", "T", "RH", "border"])

# Import particle data from the Segmentation model
particle_dataset_path = os.path.join(script_dir, "snow_crystal_segmentation/scs_out/particles.txt")

particle_dataset = pd.read_csv(particle_dataset_path, delimiter=' ')
particle_dataset["time"] = pd.to_datetime(particle_dataset["Particle_ID"].str[:15], format="%Y%m%d_%H%M%S")

# Import PTU and ETAG data
ptu_dataset_path = os.path.join(script_dir, "classification/ptu_tot.txt")
ptu_dataset = pd.read_csv(ptu_dataset_path, delimiter="\t", names=['time_og', 'P', 'height', 'T', 'RH'])
# convert the date format
ptu_dataset["time_dt"] = pd.to_datetime(ptu_dataset["time_og"], format="%Y-%m-%d %H:%M:%S")
ptu_dataset["time"] = ptu_dataset["time_dt"].dt.strftime('%Y%m%d_%H%M%S')


# WALK THORUGH THE FOLDER FILE-BY FILE
for root, dirs, files in os.walk(crystal_path):
    # Sort the files to alphabetic order
    sorted_files = sorted(files)
    for filename in sorted_files:
        file_path = os.path.join(root, filename)
        
        ## CREATE THE IMAGE
        image = mpimg.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = tf.keras.utils.load_img(file_path, target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        print(filename)

        # Run the model on it
        if image.shape[0] > 10 and image.shape[1] > 10 :
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            plt.imshow(image)
            
          # Write the top 3 results
            top_indices = np.argsort(score)[::-1][:3]
            top_classes = [class_names[i] for i in top_indices]
            top_confidences = [score[i] for i in top_indices]
            title = ", ".join(["{} ({:.2f}%)".format(c, 100 * s) for c, s in zip(top_classes, top_confidences)])
            plt.title(title)
            save_path = save_dir + filename
            save_path2 = save_dir + "circle_" + filename
            plt.savefig(save_path)
            plt.close()

          # Find the calculated diameters in particle_dataset
            matching_particle_row = particle_dataset[particle_dataset['Particle_ID'] == filename[9:-4]]
          # extract the row if found
            if not matching_particle_row.empty:
                extracted_particle_row = matching_particle_row.values[0]
            aspect_ratio = extracted_particle_row[1]
            max_dia = extracted_particle_row[5]
            min_circle = extracted_particle_row[6]
            x = extracted_particle_row[7]
            y = extracted_particle_row[8]
            border = extracted_particle_row[9]

            mask = particle_mask_path + "particle_mask" + filename[9:]
            aed = calculate_aed(mask)
            
          # find the corresponding PTU data
          # TIME CORRECTION ~10S
            time_str = filename[9:24]

            # Define the time format
            time_format = "%Y%m%d_%H%M%S"

          # Subtract 10 seconds
            new_time = datetime.strptime(time_str, time_format) - timedelta(seconds=10)
            print(new_time)
          # Convert back to string
            new_time_str = new_time.strftime(time_format)

          # Find matching rows in the dataset
            matching_ptu_row = ptu_dataset[ptu_dataset['time'] == new_time_str]

          # extract the row if found
            if not matching_ptu_row.empty:
                extracted_ptu_row = matching_ptu_row.values[0]
            p = extracted_ptu_row[1]
            height = extracted_ptu_row[2]
            t = extracted_ptu_row[3]
            rh = extracted_ptu_row[4]      
    
          # write the data to the dataframe
            data.loc[len(data)] = [
                filename[9:-4],
                class_names[np.argmax(score)],
                aed,
                min_circle,
                max_dia,
                aspect_ratio,
                x,
                y,
                p,
                height,
                t,
                rh,
                border
                ]

        else:
            print("particle too small to analyze")

## save the data
output_file = os.path.join(save_dir, "data.txt")
data.to_csv(output_file, sep="\t", index=False)
