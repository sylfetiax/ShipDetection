from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPool2D, Flatten, Dense, ZeroPadding2D, AveragePooling2D
from tensorflow.keras.models import Model


def identity_block(x, filter_size):
    x_skip = x
    x = Conv2D(filter_size, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filter_size, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    x_skip = Conv2D(filter_size, (1, 1))(x_skip)

    x = Add()([x, x_skip])
    x = Activation('relu')(x)
    return x

def convolutional_block(x, filter_size):
    x_skip = x
    x = Conv2D(filter_size, (3,3), padding='same', strides=(2,2))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Conv2D(filter_size, (3,3), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x_skip = Conv2D(filter_size, (1,1), strides=(2,2))(x_skip)
    x = Add()([x, x_skip])     
    x = Activation('relu')(x)
    return x

def ResNet_model(shape=(256, 256, 3)):
    x_input = Input(shape)
    x = ZeroPadding2D((3, 3))(x_input)
    x = Conv2D(64, kernel_size=7, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    
    block_layers = [2, 2, 2, 2]
    filter_size = 32
    
    for i in range(4):
        if i == 0:
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size)
        else:
            filter_size = filter_size*2
            x = convolutional_block(x, filter_size)
            for j in range(block_layers[i] - 1):
                x = identity_block(x, filter_size)
    
    x = AveragePooling2D((2,2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)  
    model = Model(inputs=x_input, outputs=x)
    return model
