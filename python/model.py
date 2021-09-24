import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model():
    inputs = tf.keras.Input(shape=(32,32,1), name='input')
    model = keras.Sequential()
    # model.add(inputs)
    model.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(32,32,1), padding='valid'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    # conv2d_1
    model.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(layers.Flatten())
    # dense
    model.add(layers.Dense(120, activation='tanh'))
    # dense_1
    model.add(layers.Dense(84, activation='tanh'))
    # dense_2
    model.add(layers.Dense(10, activation='sigmoid'))

    model.summary()

    # output = model(inputs)

    return model