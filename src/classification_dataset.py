from tensorflow.keras.preprocessing.image import ImageDataGenerator


def train_val_split(df, test_size, image_path, batch_size, seed=42):
    """
    Create train and validation generators for binary classification.

    Parameters:
    - df: DataFrame containing image information and class labels.
    - test_size: Proportion of the dataset to include in the validation split (float).
    - image_path: Path to the directory containing images (string).
    - batch_size: Batch size for training and validation generators (integer).
    - seed: Seed for random operations (integer).

    Returns:
    - Tuple of train and validation generators.
    """
    # Create an ImageDataGenerator with specified augmentation settings
    datagen = ImageDataGenerator(
        rescale=1./255.,
        validation_split=test_size,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.01,
        horizontal_flip=True,
        vertical_flip=True
    )

    # Create train generator
    train_generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=image_path,
        batch_size=batch_size,
        subset="training",
        target_size=(256, 256),
        class_mode="binary",
        seed=seed
    )

    # Create validation generator
    validation_generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=image_path,
        batch_size=batch_size,
        subset="validation",
        target_size=(256, 256),
        class_mode="binary",
        shuffle=True,
        sample_size=200,  # Adjust sample_size as needed
        seed=seed
    )

    return train_generator, validation_generator

    
