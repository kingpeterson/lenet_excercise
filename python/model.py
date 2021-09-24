import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import numpy as np

LAYERS_WITH_WEIGHT = ['conv2d', 'dense']

def parse_weight():
    with open('../ai_tech_challenge/lenet.json') as json_file:
        weights = json.load(json_file)
        # print(weights)
        # print(weights.keys())
    json_file.close()
    return weights.keys(), weights

def convert_str_float(weight):
    weights = weight.split(' ')
    weights = np.array(weights).astype(np.float32)
    return weights

def build_model():
    model = keras.Sequential()
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

    keys, weights = parse_weight()
    model = load_weight(model, weights)
    model.summary()


    # for layer in model.layers:
    #     print(layer.weight)
    #     print(layer.bias)
    return model

def load_weight(model, weights):
    for layer in model.layers:
        name = layer.name.split('_')[0]
        if name not in LAYERS_WITH_WEIGHT:
            continue

        number = layer.name.split('_')[-1] if '_' in layer.name else ''

        if number == '':
            weight_name = name + '_weight'
            bias_name = name + '_bias'
        else:
            weight_name = name + '_' + number + '_weight'
            bias_name = name + '_' + number + '_bias'
        
        # print(weight_name)
        # print(bias_name)
        weight = weights[weight_name]
        weight = convert_str_float(weight)

        bias = weights[bias_name]
        bias = convert_str_float(bias)
        # print(bias)

        # print(layer)
        # print(weight.shape)

        layer.weight = weight
        layer.bias = bias
        layer.bias_initializer = None
        layer.trainable = False
        
    return model