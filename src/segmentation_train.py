from tensorflow.keras.optimizers import Adam
import pandas as pd
from segmentation_dataset import *
from segmentation_model import *
from config import *
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import argparse

def train_model(model, train_gen, val_gen, epochs, steps, save_path):
    """
    Train a segmentation model using generators.

    Parameters:
    - model: Segmentation model to be trained.
    - train_gen: Generator for training data.
    - val_gen: Generator for validation data.
    - epochs: Number of epochs for training (integer).
    - steps: Number of steps per epoch (integer).
    - save_path: Path to save the trained model (string).
    """
    # Define callbacks for learning rate reduction and early stopping
    reduceLROnPlateau = ReduceLROnPlateau(
        monitor='val_dice_score',
        factor=0.33,
        patience=1,
        verbose=1,
        mode='max',
        min_delta=0.0001,
        cooldown=2,
        min_lr=1e-6
    )
    early_stopping = EarlyStopping(
        monitor="val_dice_score",
        mode="max",
        patience=5
    )

    # Set up optimizer
    optimizer = Adam(learning_rate=1e-4)

    # Compile the model with custom loss and metrics
    model.compile(optimizer=optimizer, loss=dice_binary_cross_entropy, metrics=[dice_coefficient])

    # Train the model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        steps_per_epoch=steps,
        callbacks=[reduceLROnPlateau, early_stopping],
        workers=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['dice_score'])
    plt.plot(history.history['val_dice_score'])
    plt.title('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()

    # Save the trained model
    model.save(os.path.join(save_path, 'segmentation_model.keras'))
    

def main(args):
    """
    Main function for training a U-Net model on segmentation data.

    Parameters:
    - args: Dictionary containing command-line arguments or default values.
    """
    # Set default values or use provided command-line arguments
    masks_csv = args['masks_csv'] if args['masks_csv'] else MASKS_PATH
    test_size = args['test_size'] if args['test_size'] else SEG_TEST_SIZE
    images_path = args['images_path'] if args['images_path'] else TRAIN_PATH
    epochs = args['epochs'] if args['epochs'] else SEG_EPOCHS
    batch_size = args['batch_size'] if args['batch_size'] else SEG_BATCH_SIZE
    scaling = args['scaling'] if args['scaling'] else SCALING
    save_path = args['save_path'] if args['save_path'] else SEG_SAVE_PATH

    # Read masks CSV and drop NaN values
    print('Loading csv...')
    df = pd.read_csv(masks_csv)
    df.dropna(inplace=True)

    # Split data into training and validation sets, create generators
    print('Splitting train/val...')
    train_gen, val_gen = train_val_split(df, test_size=test_size, images_path=images_path, scaling=scaling, batch_size=batch_size)
    steps = (1 - test_size) * df.shape[0] // batch_size

    # Create U-Net model
    model = UNet_model((IMAGE_SIZE // scaling, IMAGE_SIZE // scaling, 3))
    print(model.summary())

    # Train the model
    print('Training...')
    train_model(model, train_gen, val_gen, epochs, steps, save_path)


if __name__ == "__main__":

    # Create an argument parser
    parser = argparse.ArgumentParser()
    
    # Define command-line arguments
    parser.add_argument("--masks_csv", type=str, help='Type path to csv file with masks')
    parser.add_argument("--test_size", type=float, help='Set val size from 0 to 1')
    parser.add_argument("--images_path", type=str, help='Type path to train images')
    parser.add_argument("--epochs", type=int, help='Set number of epochs for training')
    parser.add_argument("--batch_size", type=int, help='Set batch size')
    parser.add_argument("--scaling", type=int, help='Set scaling parameter for image cropping (3, 6, 12, ...)')
    parser.add_argument("--save_path", type=str, help='Type folder path where the model will be saved')

    # Parse command-line arguments
    args = parser.parse_args()
    args = vars(args)  # Convert Namespace to dictionary

    # Call the main function with parsed arguments
    main(args)