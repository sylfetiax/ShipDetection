from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf
import gdown
import os

#global variables
IMAGE_SIZE = 768 # image size, using integer as all images are square
SCALING = 6 # grid size for cropping the image for segmentation model, in this case it's 6x6 grid
CL_MODEL_ID = '1tC_bRap1f4FlrEhDF7gBrLvmgijjTP7Y' # model ids for downloading
SEG_MODEL_ID = '1Nj7Ddr8fgS_JnHhq08jdeM9VLJmuR6YU' 
SEG_BATCH_SIZE = 8 # model batch sizes
CL_BATCH_SIZE = 8
SEG_TEST_SIZE = 0.3 # test size fraction for trainings
CL_TEST_SIZE = 0.3
TRAIN_PATH = './data/train_v2' 
TEST_PATH = './data/test_v2'
MASKS_PATH = '/data/train_ship_segmentation_v2'
SEG_EPOCHS = 20 # model training epochs
CL_EPOCHS = 20
SEG_SAVE_PATH = '.'
CL_SAVE_PATH = '.'


def dice_coefficient(y_true, y_pred, smooth=1e-5):
    """
    Compute the Dice Score coefficient.

    Parameters:
    - y_true: Ground truth segmentation mask (tensor).
    - y_pred: Predicted segmentation mask (tensor).
    - smooth: Smoothing term to avoid division by zero (float).

    Returns:
    - Dice Score coefficient (float).
    """

    # Flatten the true labels and predicted labels
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(y_pred)

    # Calculate the intersection between true and predicted labels
    intersection = K.sum(y_true_f * y_pred_f)

    # Calculate the Dice Score coefficient
    dice_coefficient = (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return dice_coefficient


def dice_binary_cross_entropy(y_true, y_pred):
    """
    Compute the combined loss of Binary Crossentropy and Dice Score.

    Parameters:
    - y_true: Ground truth binary labels (tensor).
    - y_pred: Predicted binary labels (tensor).

    Returns:
    - Combined loss of Binary Crossentropy and Dice Score (float).
    """

    # Calculate Binary Crossentropy
    bce = binary_crossentropy(y_true, y_pred)

    # Calculate Dice Score
    dice = 1 - dice_coefficient(y_true, y_pred)

    # Combine Binary Crossentropy and Dice Score to get the final loss
    combined_loss = bce + dice

    return combined_loss


def f1_score(y_true, y_pred):
    """
    Compute the F1 score metric for binary classification.

    Parameters:
    - y_true: Ground truth binary labels (tensor).
    - y_pred: Predicted binary labels (tensor).

    Returns:
    - F1 score (float).
    """

    # Calculate True Positives (TP), Actual Positives, and Predicted Positives
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    # Calculate recall, precision, and F1 score with handling for division by zero
    recall = TP / (Positives + K.epsilon())
    precision = TP / (Pred_Positives + K.epsilon())
    f1 = 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    return f1


def load_model_from_gdrive(id, destination_path):
    """
    Download a model from Google Drive if it doesn't exist.

    Parameters:
    - url: Google Drive URL of the model.
    - destination_path: Local path where the model will be saved.

    Returns:
    - None
    """
    if not os.path.exists(destination_path):
        # Download the model using gdown
        gdown.download(id=id, output=destination_path, quiet=False)
        print("Model has been loaded")
    else:
        # Print a message if the model file already exists
        print(f"Model '{destination_path}' already exists.")