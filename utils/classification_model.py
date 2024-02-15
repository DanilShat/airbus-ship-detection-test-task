import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def create_classification_model(num_classes=2):
    """
    This function creates a classification model using the pre-trained ResNet50 model and custom layers.

    Parameters:
    num_classes (int): The number of classes for the classification task. Default is 2.

    Returns:
    model (keras.Model): The classification model.
    """
    # Load the pre-trained ResNet50 model (excluding the top layers)
    base_model = ResNet50(weights='imagenet', include_top=False)

    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_classes, activation='sigmoid')(x)

    # This creates a model that includes the base model and the custom layers
    model = Model(inputs=base_model.input, outputs=predictions)

    return model
