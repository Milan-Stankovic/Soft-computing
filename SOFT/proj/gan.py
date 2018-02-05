from keras.models import Sequential
from keras.layers import Activation, Dense, Reshape, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.layers.advanced_activations import LeakyReLU


def build_generator(input_dim, output_size):

    model = Sequential()
    model.add(Dense(256, input_dim=input_dim))
    unit_size = 128 * output_size[0] // 8 * output_size[1] // 8
    model.add(Dense(unit_size))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    shape = (output_size[0] // 8, output_size[1] // 8, 128)
    model.add(Reshape(shape))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(3, (5, 5), padding='same'))
    model.add(Activation('sigmoid'))
    return model


def build_upsampler(input_size):

    model = Sequential()
    input_shape = (input_size[0], input_size[1], 3)
    model.add(Conv2D(32, (5, 5), padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))
    unit_size = 128 * input_size[0] // 4 * input_size[1] // 4
    model.add(Dense(unit_size))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    shape = (input_size[0] // 4, input_size[1] // 4, 128)
    model.add(Reshape(shape))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(3, (5, 5), padding='same'))
    model.add(Activation('sigmoid'))
    return model


def build_discriminator(input_size):

    model = Sequential()
    input_shape = (input_size[0], input_size[1], 3)
    model.add(Conv2D(32, (5, 5), padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def build_GAN(G, D):

    model = Sequential()
    model.add(G)

   # D.trainable = False
    model.add(D)
    return model