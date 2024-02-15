import numpy as np
import pandas as pd
from utils.unet_model import create_unet_model
from utils.loses import dice_coef, dice_loss
from utils.preprocess import preprocess_image, rle_decode
from tensorflow.keras.metrics import Recall, Precision
import tensorflow as tf

path_to_train_csv = '/kaggle/input/airbus-ship-detection/train_ship_segmentations_v2.csv' # replace with your path to train csv
base_path_to_train_images = '/kaggle/input/airbus-ship-detection/train_v2' # replace with your path to train images directory

df = pd.read_csv(path_to_train_csv)
df = df.dropna() # for this model we will only use images containing ships

# Define a function to combine EncodedPixels
def combine_encoded_pixels(s):
    s = s.dropna()
    if s.empty:
        return np.nan
    else:
        return ' '.join(s)

# Combine EncodedPixels values for each unique ImageId
df = df.groupby('ImageId')['EncodedPixels'].apply(combine_encoded_pixels).reset_index()
df = df[:6500] # I used a stripped-down dataset due to Kaggle limitations

# Creating X_train and y_train
X_train = np.array([preprocess_image(image_name, base_path_to_train_images) for image_name in df["ImageId"]])
y_train = np.array([rle_decode(encoded_pixels) for encoded_pixels in df["EncodedPixels"]])

# Defining the model
segmentation_model = create_unet_model()

# Defining the hyperparameters and compiling the model
opt = tf.keras.optimizers.Nadam(1e-4)
metrics = [dice_coef, Recall(), Precision()]
segmentation_model.compile(loss=dice_loss, optimizer=opt, metrics=metrics)

# Fit
segmentation_model.fit(X_train, y_train, epochs=30, batch_size=10)

# Save
segmentation_model.save('segmentation_model.h5')


