import os
import random
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.preprocessing import image
import numpy as np

# Specify the path to your image folder
folder_path = r"C:\Users\nandh\Desktop\Mini Project 3\FDS\train\WASHINGMACHINE"

# Get a list of all files in the folder
all_files = os.listdir(folder_path)

# Select a random subset of files (e.g., 5 files)
selected_files = random.sample(all_files, 10)

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

for file_name in selected_files:
    file_path = os.path.join(folder_path, file_name)

    # Load and resize the image
    img = load_img(file_path, color_mode='grayscale', target_size=(112, 112))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    # Generate augmented images
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=r"C:\Users\nandh\Desktop\Mini Project 3\FDS", save_prefix=file_name.split('.')[0], save_format='jpg'):
        i += 6
        if i > 74:
            break
