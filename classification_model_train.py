import numpy as np
import pandas as pd
from utils.preprocess import preprocess_image, rle_decode
from utils.classification_model import create_classification_model
from sklearn.utils import shuffle

path_to_train_csv = '/kaggle/input/airbus-ship-detection/train_ship_segmentations_v2.csv' # replace with your path to train csv
base_path_to_train_images = '/kaggle/input/airbus-ship-detection/train_v2' # replace with your path to train images directory

df = pd.read_csv(path_to_train_csv)
df = df[:8000] # I used a stripped-down dataset due to Kaggle limitations

nan_indices = df[df['EncodedPixels'].isna()].sample(n=2500).index
non_nan_indices = df[df['EncodedPixels'].notna()].sample(n=2500).index

# Create a new DataFrame with the desired counts
df = pd.concat([df.loc[nan_indices], df.loc[non_nan_indices]])

# Creating X_train and y_train
X_train = np.array([preprocess_image(image_name, base_path_to_train_images) for image_name in df["ImageId"]])
y_train = [0 if pd.isna(x) else 1 for x in df['EncodedPixels']]

y_train = np.array(y_train)

X_train, y_train = shuffle(X_train, y_train, random_state=42)

# Defining the model
classification_model = create_classification_model()

# Defining the hyperparameters and compiling the model
classification_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit
classification_model.fit(X_train, y_train, epochs=10, batch_size=10)

# Save
classification_model.save('classification_model.h5')


