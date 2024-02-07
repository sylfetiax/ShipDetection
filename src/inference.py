from tensorflow.keras.models import load_model
import os
from config import *
import numpy as np
from skimage.io import imread
import cv2
from run_length_encoding import *
import pandas as pd
import argparse
from tqdm import tqdm

def predict_unet(img, model, scaling):
    """
    Uses the UNet model to make predictions on an image divided into patches.

    Parameters:
    - img: Input image.
    - model: Trained UNet model.
    - scaling: Scaling factor for patch division.

    Returns:
    - Prediction result on the input image.
    """
    patch_size = IMAGE_SIZE // scaling
    result = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    # Iterate over patches and get predictions
    for i in range(scaling):
        for j in range(scaling):
            # Extract a patch and make a prediction
            patch = np.expand_dims(img[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size, :], axis=0)
            patch_result = model.predict(patch, verbose=False)
            patch_result = (patch_result > 0.5).astype(np.uint8)

            # Save the result of the corresponding patch in the resulting image
            result[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = patch_result[0, :, :, 0]

    return result


def inference_image(image_path, resnet_model, unet_model, scaling):
    """
    Perform image inference using a ResNet model for classification and a UNet model for segmentation.

    Parameters:
    - image_path: Path to the input image.
    - resnet_model: Trained ResNet model for classification.
    - unet_model: Trained UNet model for segmentation.
    - scaling: Scaling factor for patch division in the UNet model.

    Returns:
    - RLE-encoded segmentation result or NaN if the classification confidence is below the threshold.
    """
    # Read and normalize the input image
    image = np.array(imread(image_path)) / 255.0

    # Perform classification using the ResNet model
    classification_result = resnet_model.predict(np.expand_dims(cv2.resize(image, (256, 256)), axis=0), verbose=0)[0]

    # Check if the image is classified as having a ship (confidence above 0.5)
    if classification_result[0] < 0.5:
        return ''  # Return blank value

    # Perform segmentation using the UNet model
    segmentation_result = predict_unet(image, unet_model, scaling)

    # Encode the segmentation result using Run-Length Encoding (RLE)
    return rle_encode(segmentation_result)


def save_submission(images_path, cl_model, seg_model, scaling):
    """
    Generate a submission file for a set of images using a classification model and a segmentation model.

    Parameters:
    - images_path: Path to the directory containing input images.
    - cl_model: Trained classification model.
    - seg_model: Trained segmentation (UNet) model.
    - scaling: Scaling factor for patch division in the UNet model.

    Returns:
    - None, but saves the submission file as 'submission.csv'.
    """
    # Get the list of image filenames in the specified directory
    images = os.listdir(images_path)

    # Initialize tqdm for a progress bar
    with tqdm(total=len(images), desc="Generating Submission", unit="image") as pbar:
        # Perform inference for each image and collect the encoded masks
        encoded_masks = []
        for image in images:
            encoded_masks.append(inference_image(os.path.join(images_path, image), cl_model, seg_model, scaling))
            # Update the progress bar
            pbar.update(1)

    # Create a DataFrame with ImageId and EncodedPixels columns and save it to a CSV file
    pd.DataFrame({"ImageId": images, "EncodedPixels": encoded_masks}).to_csv('submission.csv', index=False)
    
    # Print a message indicating that the submission has been saved
    print('Submission saved.')


def main(args):
    """
    Main function to generate a submission using a ResNet model for classification and a UNet model for segmentation.

    Parameters:
    - args: Dictionary of command-line arguments.

    Returns:
    - None, but saves the submission file.
    """
    # Set default values or use values provided in command-line arguments
    images_path = args['images_path'] if args['images_path'] else TEST_PATH
    cl_path = args['cl_path'] if args['cl_path'] else os.path.join(CL_SAVE_PATH, 'classification_model.keras')
    seg_path = args['seg_path'] if args['seg_path'] else os.path.join(SEG_SAVE_PATH, 'segmentation_model.keras')
    scaling = args['scaling'] if args['scaling'] else SCALING

    # Define custom objects for loading models with custom metrics
    custom_objects_cl = {'f1_score': f1_score}
    custom_objects_seg = {'dice_coefficient': dice_coefficient, 'dice_binary_cross_entropy': dice_binary_cross_entropy}

    # Load models from google drive
    print('Loading models from google drive...')
    load_model_from_gdrive(id=CL_MODEL_ID, destination_path=os.path.join(CL_SAVE_PATH, 'classification_model.keras'))
    load_model_from_gdrive(id=SEG_MODEL_ID, destination_path=os.path.join(SEG_SAVE_PATH, 'segmentation_model.keras'))
                           
    # Load the ResNet and UNet models
    resnet = load_model(cl_path, custom_objects=custom_objects_cl)
    unet = load_model(seg_path, custom_objects=custom_objects_seg)

    # Generate and save the submission using the loaded models
    print('Processing submission...')
    save_submission(images_path, resnet, unet, scaling)


if __name__ == '__main__':
    # Create an ArgumentParser object to handle command-line arguments
    parser = argparse.ArgumentParser()
    
    # Define command-line arguments
    parser.add_argument("--images_path", type=str, help='Path to test images directory')
    parser.add_argument("--cl_path", type=str, help='Path to the classification model')
    parser.add_argument("--seg_path", type=str, help='Path to the segmentation model')
    parser.add_argument("--scaling", type=int, help='Scaling factor used in segmentation model training: 3, 6, 12, ...')

    # Parse command-line arguments
    args = parser.parse_args()
    args = vars(args)  # Convert the Namespace object to a dictionary

    # Call the main function with the parsed arguments
    main(args)

