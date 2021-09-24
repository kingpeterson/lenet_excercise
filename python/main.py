from model import build_model
import cv2
import argparse
import json
import numpy as np

LAYERS_WITH_WEIGHT = ['conv2d', 'dense']

parser = argparse.ArgumentParser()
parser.add_argument("-i", "-input", help="input img", dest="input")

args = parser.parse_args()
img_dir = args.input

def load_weight():
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


model = build_model()
keys, weights = load_weight()

# load weight
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

    layer.weight = weight
    layer.bias = bias


img = cv2.imread(img_dir, 0)
img = np.expand_dims(img, axis = -1)
img = np.expand_dims(img, axis = 0)
prediction = list(model.predict(img)[0])
print(prediction)
predicted_number = prediction.index(max(prediction))
print(predicted_number)