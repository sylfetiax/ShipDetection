import pandas as pd
from classification_dataset import *
from config import *
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from classification_model import *
import matplotlib.pyplot as plt
import os
import argparse

def train_model(model, train_gen, val_gen, epochs, steps, save_path):
    """
    Train a binary classification model using F1 score as a metric.

    Parameters:
    - model: Binary classification model to be trained.
    - train_gen: Generator for training data.
    - val_gen: Generator for validation data.
    - epochs: Number of epochs for training (integer).
    - steps: Number of steps per epoch (integer).
    - save_path: Path to save the trained model (string).
    """
    # Define callbacks for learning rate reduction and early stopping
    reduceLROnPlate = ReduceLROnPlateau(
        monitor='val_f1_score',
        factor=0.33,
        patience=1,
        verbose=1,
        mode='max',
        min_delta=0.0001,
        cooldown=2,
        min_lr=1e-6
    )
    early_stopping = EarlyStopping(
        monitor="val_f1_score",
        mode="max",
        patience=5
    )

    # Compile the model with Adam optimizer, binary crossentropy loss, and F1 score metric
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[f1_score])

    # Train the model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        steps_per_epoch=steps,
        callbacks=[reduceLROnPlate, early_stopping]
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

    # Plot training & validation F1 score
    plt.subplot(1, 2, 2)
    plt.plot(history.history['f1_score'])
    plt.plot(history.history['val_f1_score'])
    plt.title('F1 score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 score')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()

    # Save the trained model
    model.save(os.path.join(save_path, 'classification_model.keras'))


def main(args):
    """
    Main function for training a ResNet model on binary classification data.

    Parameters:
    - args: Dictionary containing command-line arguments or default values.
    """
    # Set default values or use provided command-line arguments
    masks_csv = args['masks_csv'] if args['masks_csv'] else MASKS_PATH
    test_size = args['test_size'] if args['test_size'] else CL_TEST_SIZE
    images_path = args['images_path'] if args['images_path'] else TRAIN_PATH
    epochs = args['epochs'] if args['epochs'] else CL_EPOCHS
    batch_size = args['batch_size'] if args['batch_size'] else CL_BATCH_SIZE
    save_path = args['save_path'] if args['save_path'] else CL_SAVE_PATH

    # Read masks CSV and preprocess data
    print('Loading csv...')
    df = pd.read_csv(masks_csv)
    df.drop_duplicates(subset=['ImageId'])
    df['class'] = df['EncodedPixels'].map(lambda pixels: 1 if isinstance(pixels, str) else 0)
    df_ships = df[df['class'] == 1]
    df_empty = df[df['class'] == 0]
    df = pd.concat([df_ships, df_empty.sample(df_ships.shape[0])], ignore_index=True)
    df = df.rename(columns={'ImageId': 'filename'})
    df['class'] = df['class'].astype(str)
    df.sample(frac=1).reset_index(drop=True)

    # Create train and validation generators
    print('Splitting train/val...')
    train_gen, val_gen = train_val_split(df, test_size, images_path, batch_size)
    steps = (1 - test_size) * df.shape[0] // batch_size

    # Define ResNet model
    model = ResNet_model()
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
    parser.add_argument("--save_path", type=str, help='Type folder path where the model will be saved')

    # Parse command-line arguments
    args = parser.parse_args()
    args = vars(args)  # Convert Namespace to dictionary

    # Call the main function with parsed arguments
    main(args)