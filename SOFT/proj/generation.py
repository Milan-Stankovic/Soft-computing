import numpy as np


def generate(G, source):

    input_dim = G.input_shape[1]
    images = G.predict(source)
    images = images * 255
    return images