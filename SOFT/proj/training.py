import math
import numpy as np


def display(step, steps):

    print('Step: '+str(step+1)+'/'+str(steps), end='')



def train(G, D, GAN, sets, batch):


    np.random.shuffle(sets)
    input_dim = G.input_shape[1]

    steps = math.ceil(len(sets) / batch)
    for step in range(steps):

        real = sets[step*batch:(step+1)*batch]
        samples = len(real)

        answer = np.ones(samples)
        D_loss = D.train_on_batch(x=real, y=answer)

        noise = np.random.uniform(0, 1, size=(samples, input_dim))
        generated = G.predict(noise)
        answer = np.zeros(samples)
        D_loss = D.train_on_batch(x=generated, y=answer)

        noise = np.random.uniform(0, 1, size=(samples, input_dim))
        answer = np.ones(samples)
        GAN_loss = GAN.train_on_batch(x=noise, y=answer)

        display(step, steps)
    print()
    return (D_loss, GAN_loss)


def train_with_images(G_before, G, D, GAN, sets, batch):


    np.random.shuffle(sets)
    input_dim = G_before.input_shape[1]

    steps = math.ceil(len(sets) / batch)
    for step in range(steps):

        real = sets[step*batch:(step+1)*batch]
        samples = len(real)

        answer = np.ones(samples)
        D_loss = D.train_on_batch(x=real, y=answer)

        noise = np.random.uniform(0, 1, size=(samples, input_dim))
        G_out = G_before.predict(noise)
        generated = G.predict(G_out)
        answer = np.zeros(samples)
        D_loss = D.train_on_batch(x=generated, y=answer)

        noise = np.random.uniform(0, 1, size=(samples, input_dim))
        G_out = G_before.predict(noise)
        answer = np.ones(samples)
        GAN_loss = GAN.train_on_batch(x=G_out, y=answer)

        display(step, steps)
    print()
    return (D_loss, GAN_loss)