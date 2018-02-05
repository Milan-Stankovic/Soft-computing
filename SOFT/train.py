from argparse import ArgumentParser

import numpy as np
from keras.optimizers import Adam

from proj.file import save_model
from proj.gan import build_generator, build_upsampler, build_discriminator, build_GAN
from proj.generation import generate
from proj.image import save_images, load_images, to_dirname
from proj.training import train, train_with_images


def get_args():
    parser = ArgumentParser('GAN')
    parser.add_argument('-d', '--dim', type=int, default=100) #Dimenzija
    parser.add_argument('-z', '--size', type=int, nargs=2, default=[64, 64]) #Velicina slike
    parser.add_argument('-b', '--batch', type=int, default=64) #Batch size
    parser.add_argument('-e', '--epoch', type=int, default=3000) #Epohe
    parser.add_argument('-s', '--save', type=int, default=100) #Kada pamti
    parser.add_argument('-i', '--input', type=str, default='data') #Data set
    parser.add_argument('-o', '--output', type=str, default='save') #Result set
    return parser.parse_args()


def main():
    args = get_args()

    input_dim = args.dim
    image_size2x = args.size
    image_size = (image_size2x[0]//2, image_size2x[1]//2)
    batch = args.batch
    epochs = args.epoch
    save_freq = args.save
    input_dirname = to_dirname(args.input)
    output_dirname = to_dirname(args.output)
    # Stage 1
    G1 = build_generator(input_dim=input_dim, output_size=image_size)
    D1 = build_discriminator(input_size=image_size)
    optimizer = Adam(lr=1e-5, beta_1=0.1)
    D1.compile(loss='mean_squared_error', optimizer=optimizer)
    GAN1 = build_GAN(G1, D1)
    optimizer = Adam(lr=1e-4, beta_1=0.5)
    GAN1.compile(loss='mean_squared_error', optimizer=optimizer)
    # Stage 2
    G2 = build_upsampler(input_size=image_size)
    D2 = build_discriminator(input_size=image_size2x)
    optimizer = Adam(lr=1e-5, beta_1=0.1)
    D2.compile(loss='mean_squared_error', optimizer=optimizer)
    GAN2 = build_GAN(G2, D2)
    optimizer = Adam(lr=1e-4, beta_1=0.5)
    GAN2.compile(loss='mean_squared_error', optimizer=optimizer)

    save_model(G1, 'G1_model.json')
    save_model(D1, 'D1_model.json')
    save_model(G2, 'G2_model.json')
    save_model(D2, 'D2_model.json')

    images = load_images(name=input_dirname, size=image_size)
    images2x = load_images(name=input_dirname, size=image_size2x)

    for epoch in range(epochs):
        # Stage 1
        print('Epoch: '+str(epoch+1)+'/'+str(epochs)+' - Stage: 1')
        train(G1, D1, GAN1, sets=images, batch=batch)
        # Stage 2
        print('Epoch: '+str(epoch+1)+'/'+str(epochs)+' - Stage: 2')
        train_with_images(G1, G2, D2, GAN2, sets=images2x, batch=batch)
        if (epoch + 1) % save_freq == 0:

            noise = np.random.uniform(0, 1, (batch, input_dim))
            results1 = generate(G1, source=noise)
            save_images(results1, name=output_dirname+'stage1/'+str(epoch+1))
            results2 = generate(G2, source=results1/255)
            save_images(results2, name=output_dirname+'stage2/'+str(epoch+1))
            G1.save_weights('G1_weights.hdf5')
            D1.save_weights('D1_weights.hdf5')
            G2.save_weights('G2_weights.hdf5')
            D2.save_weights('D2_weights.hdf5')


if __name__ == '__main__':
    main()