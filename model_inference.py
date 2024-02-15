from utils.unet_model import create_unet_model
from utils.classification_model import create_classification_model
from utils.loses import dice_coef, dice_loss
from utils.preprocess import preprocess_image, rle_decode, rle_encode

import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

custom_objects = {'dice_coef': dice_coef}
segmentation_model = tf.keras.models.load_model('/saved_models/segmentation_model.h5', custom_objects=custom_objects)
classification_model = tf.keras.models.load_model('/saved_models/classification_model.h5')

path_to_test_csv = '/kaggle/input/airbus-ship-detection/sample_submission_v2.csv' # replace with your path to test csv
base_path_to_test_images = '/kaggle/input/airbus-ship-detection/test_v2' # replace with your path to test images directory

def process_images(df, base_path, show_results=False):
    # Initialize an empty list to store the encoded pixels
    encoded_pixels = []

    # Iterate over each image
    for image_name in df['ImageId']:
        # Preprocess image
        image = preprocess_image(image_name, base_path)
        image = np.expand_dims(image, axis=0)

        # Predict with classification model
        pred = classification_model.predict(image)

        if pred < 0.5:
            if show_results:
                print('No ships were found.')
            # Create an empty mask
            mask = np.zeros_like(image)
        else:
            # Predict with segmentation model
            mask = segmentation_model.predict(image)
            if show_results:
                print('Mask was predicted.')

        if show_results:
            # Show image and mask
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(np.squeeze(image, axis = 0))
            ax[0].set_title('Image')
            ax[1].imshow(np.squeeze(mask, axis=0))
            ax[1].set_title('Mask')
            plt.show()

        # Append the result of rle_encode function on prediction to the list
        encoded_pixels.append(rle_encode(np.squeeze(mask, axis = 0)))

    # Add the encoded pixels as a new column to the DataFrame
    df['EncodedPixels'] = encoded_pixels

sub_df = pd.read_csv(path_to_test_csv)
sub_path = base_path_to_test_images
process_images(sub_df, sub_path)


