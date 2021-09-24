from model import build_model, load_weight
import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-i", "-input", help="input img", dest="input")

args = parser.parse_args()
img_dir = args.input

model = build_model()

img = cv2.imread(img_dir, 0)
img = np.expand_dims(img, axis = -1)
img = np.expand_dims(img, axis = 0)
prediction = list(model.predict(img)[0])
print('Resulting vector:')
for i, each in enumerate(prediction):
    print('{}    {}'.format(i, round(float(each), 4)))
predicted_number = prediction.index(max(prediction))
print('\nClassification: {}'.format(predicted_number))