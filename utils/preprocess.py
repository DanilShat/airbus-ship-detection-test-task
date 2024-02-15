import numpy as np
import pandas as pd
import cv2

def preprocess_image(image_name, base_path, target_size=(224, 224)):
    """
    This function preprocesses an image for model input.

    Parameters:
    image_name (str): The name of the image file.
    base_path (str): The directory path where the image file is located.
    target_size (tuple): The desired dimensions to resize the image to. Default is (224, 224).

    Returns:
    image (numpy.ndarray): The preprocessed image.
    """
    # Construct full image path
    image_path = f"{base_path}/{image_name}"  # Update the path as needed
    
    # Load image
    image = cv2.imread(image_path)
    
    # Check if image was loaded successfully
    if image is None:
        print(f"Failed to load image at {image_path}")
        return None
    
    # Convert color space from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to target size
    image = cv2.resize(image, target_size)
    
    # Normalize pixel values
    image = image / 255.0
    
    return image


def rle_decode(mask_rle, shape=(768, 768), new_shape=(224, 224)):
    """
    This function decodes a run-length encoded (RLE) mask and resizes it.

    Parameters:
    mask_rle (str): The run-length encoded mask, formatted as (start length).
    shape (tuple): The original (height, width) of the mask to return. Default is (768, 768).
    new_shape (tuple): The desired (height, width) of the resized mask. Default is (224, 224).

    Returns:
    img (numpy.ndarray): The decoded and resized mask. The mask is 1, and the background is 0.
    """
    # If the mask is empty (i.e., no object), return a zero-filled mask
    if pd.isna(mask_rle):
        return np.zeros(new_shape, dtype=np.uint8)
    
    # Split the RLE string into starts and lengths
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    
    # Adjust the starts (Python is 0-indexed)
    starts -= 1
    
    # Calculate the ends of the runs
    ends = starts + lengths
    
    # Initialize an empty image
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    
    # Fill in the runs
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    # Reshape the image and transpose it (RLE is often transposed)
    img = img.reshape(shape).T
    
    # Resize the image to the desired shape
    img = cv2.resize(img, new_shape)
    
    # Return the image, ensuring it's in the right data type
    return img.astype(np.uint8)



def rle_encode(img, shape=(768, 768), new_shape=(224, 224), threshold=0.4):
    """
    This function encodes a binary mask using run-length encoding (RLE) and resizes it.

    Parameters:
    img (numpy.ndarray): A 2D array of shape (height, width).
    shape (tuple): The original (height, width) of the mask. Default is (768, 768).
    new_shape (tuple): The desired (height, width) of the resized mask. Default is (224, 224).
    threshold (float): A value to binarize the input array. Default is 0.4.

    Returns:
    str: The run-length encoded mask, formatted as (start length).
    """
    # Binarize the input array based on the given threshold
    img = np.where(img > threshold, 1, 0)
    
    # Resize the binary image to the original shape
    img = cv2.resize(img, shape, interpolation=cv2.INTER_NEAREST)
    
    # Flatten the image and transpose it
    pixels = img.T.flatten()
    
    # Add zeros at the beginning and end of the flattened array
    pixels = np.concatenate([[0], pixels, [0]])
    
    # Find the indices where the pixel values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    
    # Calculate the lengths of the runs
    runs[1::2] -= runs[::2]
    
    # Return the RLE string
    return ' '.join(str(x) for x in runs)


