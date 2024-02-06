from config import *
from run_length_encoding import *
import numpy as np
from skimage.io import imread
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def index_to_coordinates(index, step):
    """
    Convert a linear index to corresponding row and column coordinates.

    Parameters:
    - index: Linear index (integer).
    - step: Step size for indexing (integer).

    Returns:
    - Tuple of row and column coordinates.
    """
    # Calculate the column coordinate
    col = index // IMAGE_SIZE

    # Calculate the row coordinate
    row = index % IMAGE_SIZE

    # Adjust coordinates based on the step size
    row //= step
    col //= step

    return row, col


def crop(im, mask, row, col, step):
    """
    Crop an image and its corresponding mask based on row, column, and step size.

    Parameters:
    - im: Image to be cropped (numpy array).
    - mask: Mask corresponding to the image (numpy array).
    - row: Row coordinate for cropping.
    - col: Column coordinate for cropping.
    - step: Step size for cropping (integer).

    Returns:
    - Tuple of cropped image and mask.
    """
    # Calculate the start and end indices for row and column based on the step size
    s_row_start = row * step
    s_row_end = s_row_start + step

    s_col_start = col * step
    s_col_end = s_col_start + step

    # Crop the image and mask
    cropped_im = im[s_row_start:s_row_end, s_col_start:s_col_end]
    cropped_mask = mask[s_row_start:s_row_end, s_col_start:s_col_end]

    return cropped_im, cropped_mask


def make_image_gen(in_df, images_path, scaling, batch_size):
    """
    Generate batches of images and their corresponding masks for training.

    Parameters:
    - in_df: DataFrame containing image information.
    - images_path: Path to the directory containing images (string).
    - scaling: Scaling factor for random cropping (integer).
    - batch_size: Batch size for training (integer).

    Yields:
    - Tuple of batched images and masks.
    """
    all_batches = list(in_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    step = IMAGE_SIZE // scaling
    while True:
        np.random.shuffle(all_batches)

        for c_img_id, c_masks in all_batches:
            # Load the RGB image
            rgb_path = os.path.join(images_path, c_img_id)
            c_img = imread(rgb_path)

            # Decode masks and handle random cropping
            enc = c_masks['EncodedPixels'].values
            c_mask = masks_as_image(enc)
            
            if isinstance(enc[0], float):
                grid_pos = np.random.randint(0, scaling, size=2)
            else:
                grid_pos = index_to_coordinates(int(enc[0].split()[2]), step)

            c_img, c_mask = crop(c_img, c_mask, grid_pos[0], grid_pos[1], step)

            # Append the cropped image and mask to the output lists
            out_rgb.append(c_img)
            out_mask.append(c_mask)

            # Yield batches when the batch size is reached
            if len(out_rgb) >= batch_size:
                yield np.stack(out_rgb, 0) / 255.0, np.stack(out_mask, 0)
                out_rgb, out_mask = [], []


# Data augmentation parameters
dg_args = dict(rotation_range = 10, 
                  width_shift_range = 0.1, 
                  height_shift_range = 0.1, 
                  shear_range = 0.01, 
                  horizontal_flip = True, 
                  vertical_flip = True,
                  fill_mode = 'reflect')

# Create ImageDataGenerator for images
image_gen = ImageDataGenerator(**dg_args)

# Create ImageDataGenerator for masks
label_gen = ImageDataGenerator(**dg_args)


def create_aug_gen(in_gen):
    """
    Create an augmented generator for input images and masks.

    Parameters:
    - in_gen: Original data generator.

    Yields:
    - Tuple of augmented images and masks.
    """
    for in_x, in_y in in_gen:
        # Use a random seed to keep augmentation synchronized between images and masks
        seed = np.random.choice(range(1000))

        # Generate augmented images
        g_x = image_gen.flow(255 * in_x,
                             batch_size=in_x.shape[0],
                             seed=seed,
                             shuffle=True)

        # Generate augmented masks
        g_y = label_gen.flow(in_y,
                             batch_size=in_x.shape[0],
                             seed=seed,
                             shuffle=True)

        # Yield augmented images and masks
        yield next(g_x) / 255.0, next(g_y)


def train_val_split(df, test_size, images_path, scaling, batch_size, random_state=42):
    """
    Perform train-validation split on a DataFrame and create augmented generators.

    Parameters:
    - df: DataFrame containing image information.
    - test_size: Proportion of the dataset to include in the validation split (float).
    - random_state: Seed for random operations (integer).

    Returns:
    - Tuple of training and validation generators.
    """
    # Perform train-validation split
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # Create augmented generator for the training set
    train_gen = create_aug_gen(make_image_gen(train_df, scaling=scaling, images_path=images_path, batch_size=batch_size))

    # Create a non-augmented generator for the validation set
    val_gen = next(make_image_gen(val_df,scaling=scaling, images_path=images_path, batch_size=200))

    return train_gen, val_gen