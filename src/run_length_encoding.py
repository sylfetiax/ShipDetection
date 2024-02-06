import numpy as np
from skimage.morphology import label


def rle_encode(img):
    """
    Run-Length Encoding (RLE) for binary masks.

    Parameters:
    - img: Binary image mask (numpy array).

    Returns:
    - Encoded RLE string.
    """
    # Flatten the image and add zeros at the beginning and end
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])

    # Identify runs where pixel values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths of runs
    runs[1::2] -= runs[::2]

    # Convert runs to a string and join them
    encoded_rle = ' '.join(str(x) for x in runs)

    return encoded_rle


def rle_decode(mask_rle, shape=(768, 768)):
    """
    Run-Length Decoding (RLE) for binary masks.

    Parameters:
    - mask_rle: Encoded RLE string.
    - shape: Shape of the target image (tuple), default is (768, 768).

    Returns:
    - Decoded binary image mask (numpy array).
    """
    # Split the RLE string into starts and lengths
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    # Adjust starts to be zero-based
    starts -= 1

    # Calculate ends of runs
    ends = starts + lengths

    # Initialize an array for the image
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    # Set the pixels corresponding to the runs to 1
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape the flattened image to the specified shape
    decoded_mask = img.reshape(shape).T

    return decoded_mask  


def multi_rle_encode(img):
    """
    Run-Length Encoding (RLE) for multiple connected components in a binary mask.

    Parameters:
    - img: Binary image mask (numpy array).

    Returns:
    - List of RLE encoded strings for each connected component.
    """
    # Label connected components in the binary mask
    labels = label(img[:, :, 0])

    # Encode each connected component separately and return a list of RLE strings
    return [rle_encode(labels == k) for k in np.unique(labels[labels > 0])]


def masks_as_image(in_mask_list):
    """
    Create a single binary mask array from a list of individual ship masks.

    Parameters:
    - in_mask_list: List of RLE encoded strings.

    Returns:
    - Combined binary mask array for all ships (numpy array).
    """
    # Initialize an array to store the combined mask
    all_masks = np.zeros((768, 768), dtype=np.int16)

    # Iterate through the list of RLE encoded strings
    for mask in in_mask_list:
        # Check if the mask is a string (RLE encoded)
        if isinstance(mask, str):
            # If RLE encoded, decode and add to the combined mask
            all_masks += rle_decode(mask)

    # Expand dimensions to create a 3D array (single channel)
    return np.expand_dims(all_masks, -1)