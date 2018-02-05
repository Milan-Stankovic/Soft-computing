import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def to_dirname(name):

    if name[-1:] == '/':
        return name
    else:
        return name + '/'


def check_dir(name):


    if os.path.isdir(name) == False:
        os.makedirs(name)

def save_images(images, name, ext='.jpg'):

    check_dir(name)

    images = images.astype(np.uint8)
    for i in range(len(images)):
        image = Image.fromarray(images[i])

        image.save(name+'/result'+str(i)+ext)


def load_images(name, size, ext='.jpg'):

    images = []

    for file in tqdm(os.listdir(name)):
        if os.path.splitext(file)[1] != ext:

            continue
        image = Image.open(name+file)
        if image.mode != "RGB":

            image.convert("RGB")
        image = image.resize(size)
        image = np.array(image)
        images.append(image)

    images = np.array(images)

    images = images / 255
    return images