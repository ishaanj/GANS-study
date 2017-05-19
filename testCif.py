from __future__ import print_function

from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
from PIL import Image

from six.moves import range

import keras.backend as K
from keras.datasets import cifar10
from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
import numpy as np
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

np.random.seed(1337)

K.set_image_data_format('channels_first')

latent_size = 100
imgh = 32
imgw = 32
imgc = 3

def build_generator(latent_size):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 1, 28, 28)
    cnn = Sequential()

    cnn.add(Dense(1024, input_dim=latent_size, activation='relu'))
    cnn.add(Dense(128 * 8 * 8, activation='relu'))
    cnn.add(Reshape((128, 8, 8)))

    # upsample to (..., 14, 14)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Conv2D(128, 5, padding='same',
                   activation='relu',
                   kernel_initializer='glorot_normal'))

    # upsample to (..., 28, 28)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Conv2D(128, 5, padding='same',
                   activation='relu',
                   kernel_initializer='glorot_normal'))

    # take a channel axis reduction
    cnn.add(Conv2D(3, 2, padding='same',
                   activation='tanh',
                   kernel_initializer='glorot_normal'))

    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(latent_size, ))

    # this will be our label
    image_class = Input(shape=(1,), dtype='int32')

    # 10 classes in MNIST
    cls = Flatten()(Embedding(10, latent_size,
                              embeddings_initializer='glorot_normal')(image_class))

    # hadamard product between z-space and a class conditional embedding
    h = layers.multiply([latent, cls])

    fake_image = cnn(h)

    return Model([latent, image_class], fake_image)


def build_discriminator():
    cnn = Sequential()

    cnn.add(Conv2D(64, 3, padding='same', strides=2,
                   input_shape=(3, 32, 32)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(128, 3, padding='same', strides=2))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(128, 3, padding='same', strides=2))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(256, 3, padding='same', strides=1))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())

    image = Input(shape=(3, 32, 32))

    features = cnn(image)

    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    fake = Dense(1, activation='sigmoid', name='generation')(features)
    aux = Dense(10, activation='softmax', name='auxiliary')(features)

    return Model(image, [fake, aux])

generator = build_generator(latent_size)
discriminator = build_discriminator()
generator.load_weights(
    'params_generator_epoch_049.hdf5')
discriminator.load_weights(
    'params_discriminator_epoch_049.hdf5')
noise = np.random.uniform(-1, 1, (100, latent_size))

sampled_labels = np.array([
    [i] * 10 for i in range(10)
]).reshape(-1, 1)

# get a batch to display
sample_list = generator.predict(
    [noise, sampled_labels], verbose=0)
sample_list = np.transpose(sample_list, (0, 2, 3, 1))
print(sample_list.shape)

# img = Image.fromarray(np.transpose(generated_images, (0, 2, 3, 1))[9], 'RGB')
# img.save('temp.png')
# img.show()
def plot_data_figs():
    fig_list = []
    #for samples in list(sample_list[-1]):
    fig = plt.figure(figsize=(32, 32))
    fig_list.append(fig)
    gs1 = gs.GridSpec(10, 10)
    gs1.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(sample_list):
        ax = plt.subplot(gs1[i])
        plt.axis('off')
        ax.set_aspect('equal')
        if imgc == 1:
            plt.imshow(sample.reshape(imgh, imgw), cmap ="Greys_r")
        else:
            plt.imshow(sample.reshape(imgh, imgw, imgc))
    plt.show(fig_list[-1])
    return fig_list[-1]

plot_data_figs()