import tensorflow as tf

def dice_coef(y_true, y_pred, smooth = 1e-15):
    """
    This function calculates the Dice coefficient between the true and predicted masks.

    Parameters:
    y_true (tf.Tensor): The ground truth mask.
    y_pred (tf.Tensor): The predicted mask.
    smooth (float): A small constant to avoid division by zero. Default is 1e-15.

    Returns:
    float: The Dice coefficient.
    """
    y_true = tf.cast(y_true, tf.float32)

    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    """
    This function calculates the Dice loss between the true and predicted masks.

    Parameters:
    y_true (tf.Tensor): The ground truth mask.
    y_pred (tf.Tensor): The predicted mask.

    Returns:
    float: The Dice loss.
    """
    return 1.0 - dice_coef(y_true, y_pred)