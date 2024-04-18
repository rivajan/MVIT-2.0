import os
import shutil
import random

def split_images(folder_path, train_folder, test_folder, split_ratio=0.5):
    # Create train and test folders if they don't exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # List all files in the folder
    files = os.listdir(folder_path)
    # Shuffle the list to randomize image order
    random.shuffle(files)

    # Calculate the number of images for each set
    num_train_images = int(len(files) * split_ratio)
    num_test_images = len(files) - num_train_images

    # Copy images to train and test folders
    for i, file in enumerate(files):
        src = os.path.join(folder_path, file)
        if i < num_train_images:
            dst = os.path.join(train_folder, file)
        else:
            dst = os.path.join(test_folder, file)
        shutil.copy(src, dst)

    print(f"{num_train_images} images copied to {train_folder}")
    print(f"{num_test_images} images copied to {test_folder}")

folder_path = "./Cropped_Selected_Images/"
train_folder = "./trainA/"
test_folder = "./testA/"
split_ratio = 0.5  # Half of the images will go to train and half to test

split_images(folder_path, train_folder, test_folder, split_ratio)
