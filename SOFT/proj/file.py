import os
from keras.models import model_from_json


def save_model(model, name):

    json = model.to_json()
    with open(name, 'w') as f:
        f.write(json)


def load_model(name):

    with open(name) as f:
        json = f.read()
    model = model_from_json(json)
    return model