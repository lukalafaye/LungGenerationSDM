import os
import random
from shutil import copyfile, move
from collections import defaultdict
from PIL import Image
import seaborn as sns
import pandas as pd
import numpy as np
from itertools import combinations

# Paths
data_dir = "data"
images_dir = "DATA_FOLDER"
masks_dir = "MASK_FOLDER"
train_dir = "train"
val_dir = "val"
test_dir = "test"

# Get list of all image and mask files
mask_files = [file for file in os.listdir(data_dir) if file.endswith(".png") and "_lung" in file]
image_files = [file for file in os.listdir(data_dir) if file.endswith(".png") and "_lung" not in file]

# Function to get colors from an image
def get_colors(image_path):
    with Image.open(image_path) as img:
        colors = img.getcolors()
    return [colors[i][1] for i in range(len(colors))]

# Initialize heatmap and dictionaries
heatmap = np.zeros(shape=(5,5))
color_files = defaultdict(list)
files_color = defaultdict(list)

# Populate dictionaries with mask colors
for mask_file in mask_files:
    colors = get_colors(os.path.join(data_dir, mask_file))
    files_color[mask_file] = colors
    for color in colors:
        color_files[color].append(mask_file)
        for c1 in colors:
            for c2 in colors:
                heatmap[c1][c2] += 1

# Create label combinations dictionary
label_combinations = defaultdict(list)
for mask_file, colors in files_color.items():
    sorted_colors = sorted(colors)
    label_combinations[(tuple(sorted_colors),)].append(mask_file)

# Shuffle the files for each combination
for comb in label_combinations.keys():
    random.shuffle(label_combinations[comb])

# Convert the defaultdict to a regular dictionary
label_combinations_dict = dict(label_combinations)

# Function to get image file from mask file
def image_from_mask(mask_file):
    image_file = mask_file.replace("_lung", "")
    return image_file

# Calculate the number of samples for each combination in train, val, and test sets
train_mask_files = []
val_mask_files = []
test_mask_files = []

for comb in label_combinations_dict.keys():
    total_samples = len(label_combinations_dict[comb])
    train_samples = int(0.7 * total_samples)
    val_samples = int(0.15 * total_samples)
    test_samples = total_samples - train_samples - val_samples

    train_mask_files.extend(label_combinations_dict[comb][:train_samples])
    val_mask_files.extend(label_combinations_dict[comb][train_samples:train_samples + val_samples])
    test_mask_files.extend(label_combinations_dict[comb][train_samples + val_samples:])

# Create directories for images and masks
os.makedirs(os.path.join(images_dir, train_dir), exist_ok=True)
os.makedirs(os.path.join(images_dir, val_dir), exist_ok=True)
os.makedirs(os.path.join(images_dir, test_dir), exist_ok=True)
os.makedirs(os.path.join(masks_dir, train_dir), exist_ok=True)
os.makedirs(os.path.join(masks_dir, val_dir), exist_ok=True)
os.makedirs(os.path.join(masks_dir, test_dir), exist_ok=True)

# Function to copy and rename files
def copy_and_rename_files(mask_files, prefix, images_dir, masks_dir, train_dir, val_dir, test_dir):
    for i, mask_file in enumerate(mask_files):
        image_file = image_from_mask(mask_file)
        new_image_name = f"{prefix}_{i+1}.png"
        new_mask_name = f"{prefix}_{i+1}.png"
        
        if prefix == 'tr':
            copyfile(os.path.join(data_dir, image_file), os.path.join(images_dir, train_dir, new_image_name))
            copyfile(os.path.join(data_dir, mask_file), os.path.join(masks_dir, train_dir, new_mask_name))
        elif prefix == 'val':
            copyfile(os.path.join(data_dir, image_file), os.path.join(images_dir, val_dir, new_image_name))
            copyfile(os.path.join(data_dir, mask_file), os.path.join(masks_dir, val_dir, new_mask_name))
        elif prefix == 'ts':
            copyfile(os.path.join(data_dir, image_file), os.path.join(images_dir, test_dir, new_image_name))
            copyfile(os.path.join(data_dir, mask_file), os.path.join(masks_dir, test_dir, new_mask_name))

# Copy and rename files for train, val, and test sets
copy_and_rename_files(train_mask_files, 'tr', images_dir, masks_dir, train_dir, val_dir, test_dir)
copy_and_rename_files(val_mask_files, 'val', images_dir, masks_dir, train_dir, val_dir, test_dir)
copy_and_rename_files(test_mask_files, 'ts', images_dir, masks_dir, train_dir, val_dir, test_dir)

print("Train Annotations Directory: ", len([name for name in os.listdir(os.path.join(masks_dir, train_dir))]))
print("Train Images Directory: ", len([name for name in os.listdir(os.path.join(images_dir, train_dir))]))
print("Validation Annotations Directory: ", len([name for name in os.listdir(os.path.join(masks_dir, val_dir))]))
print("Validation Images Directory: ", len([name for name in os.listdir(os.path.join(images_dir, val_dir))]))
print("Test Annotations Directory: ", len([name for name in os.listdir(os.path.join(masks_dir, test_dir))]))
print("Test Images Directory: ", len([name for name in os.listdir(os.path.join(images_dir, test_dir))]))
