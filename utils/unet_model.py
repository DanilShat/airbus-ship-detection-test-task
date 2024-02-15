from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2

def create_unet_model():
    """
    This function creates a U-Net model using the pre-trained MobileNetV2 model and custom layers.

    Returns:
    model (keras.Model): The U-Net model.
    """
    # Define the input shape
    inputs = Input(shape=(224, 224, 3), name="input_image")
    
    # Load the MobileNetV2 model for the encoder part of the U-Net
    encoder = MobileNetV2(input_tensor=inputs, weights="imagenet", include_top=False, alpha=0.35)
    
    # Define the names of the layers to be used for skip connections
    skip_connection_names = ["input_image", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu"]
    
    # Get the output of the last layer to be used for skip connections
    encoder_output = encoder.get_layer("block_13_expand_relu").output
    
    # Define the number of filters for the convolutional layers
    f = [16, 32, 48, 64]
    
    # Start with the encoder output
    x = encoder_output
    
    # Loop over the skip connection layers
    for i in range(1, len(skip_connection_names)+1, 1):
        # Get the output of the skip connection layer
        x_skip = encoder.get_layer(skip_connection_names[-i]).output
        
        # Upsample the previous layer output
        x = UpSampling2D((2, 2))(x)
        
        # Concatenate the upsampled output and the skip connection output
        x = Concatenate()([x, x_skip])
        
        # Apply two convolutional layers with batch normalization and ReLU activation
        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
    # Apply a final convolutional layer with sigmoid activation
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)
    
    # Create the model
    model = Model(inputs, x)
    
    return model
