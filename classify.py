import cv2
import numpy as np
from tensorflow import keras
from configurations.config import IM_SIZE


def get_prep_img(frame):
    list_item = []
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IM_SIZE)
    list_item.append(img)
    np_arr = np.array(list_item, dtype='float32')
    return np_arr


def predict_safety(img_arr, model):
    predictions = model.predict(img_arr)
    predicted_labels = np.argmax(predictions, axis=1)
    if predicted_labels[0] == 1:
        return 'SAFE'
    else:
        return 'UNSAFE'

