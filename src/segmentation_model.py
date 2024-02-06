from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model


def conv_block(x, filters, kernel_size=3, padding='same'):
    x = Conv2D(filters, kernel_size, activation='relu', padding=padding)(x)
    x = Conv2D(filters, kernel_size, activation='relu', padding=padding)(x)
    return x

def enc_block(x, filters):
    x = conv_block(x, filters)
    pool = MaxPooling2D(pool_size=(2, 2))(x)
    return x, pool 

def dec_block(x, skip, filters, kernel_size=3, padding='same'):
    x = UpSampling2D(size=(2, 2))(x)
    x = concatenate([x, skip], axis=-1)
    x = conv_block(x, filters, kernel_size, padding)
    return x

def UNet_model(input_shape, filters=32):
    inputs = Input(shape=input_shape)
    # Encoder
    conv1, pool1 = enc_block(inputs, filters)  
    conv2, pool2 = enc_block(pool1, filters * 2)
    conv3, pool3 = enc_block(pool2, filters * 4)
    # Bottleneck
    conv4 = conv_block(pool3, filters * 8)
    # Decoder
    up5 = dec_block(conv4, conv3, filters * 4)
    up6 = dec_block(up5, conv2, filters * 2)
    up7 = dec_block(up6, conv1, filters)
    # Output layer
    output = Conv2D(1, 1, activation='sigmoid')(up7)

    model = Model(inputs=inputs, outputs=output)
    return model