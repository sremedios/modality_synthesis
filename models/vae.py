from keras.layers import Input, Dense, Conv3D, MaxPooling3D, UpSampling3D,\
    Flatten, Lambda, Reshape, GlobalMaxPooling3D, Conv3DTranspose,\
    AveragePooling3D
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,\
    Flatten, Lambda, Reshape, GlobalMaxPooling2D, Conv2DTranspose,\
    AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import metrics
import tensorflow as tf

from sklearn.metrics import normalized_mutual_info_score
from utils.load_data import pad_image
import matplotlib.pyplot as plt
import numpy as np
import json


def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm, ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)


BETA = 1


def plot_latent_sampling(models,
                         dims,
                         batch_size=128):

    encoder, decoder = models

    # display a nxn 2D manifold of brains
    n = 10

    t_dims = (dims[1], dims[0])
    figure = np.zeros((t_dims[0]* n, t_dims[1]* n))
    figure_name_space = "latent_space_sampling.png"
    # linearly spaced coordinates corresponding to the 2D plot
    # of brain classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            brain = x_decoded[0].reshape(dims[0], dims[1])
            # scale for plotting
            brain = (brain/np.max(brain) * 255).astype(np.uint8)
            figure[i * t_dims[0]: (i + 1) * t_dims[0],
                   j * t_dims[1]: (j + 1) * t_dims[1]] = brain.T

    plt.figure(figsize=(10, 10))
    start_range = t_dims[0] // 2
    end_range = n * t_dims[0]+ start_range + 1
    pixel_range_x = np.arange(start_range, end_range, t_dims[0])
    pixel_range_y = np.arange(start_range, end_range, t_dims[1])
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range_x, sample_range_x)
    plt.yticks(pixel_range_y, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(figure_name_space)
    plt.show()


def inception_module_2D(prev_layer, ds=2):
    a = Conv2D(64//ds, (1, 1), strides=(1, 1), padding='same')(prev_layer)

    b = Conv2D(96//ds, (1, 1), strides=(1, 1),
               activation='relu', padding='same')(prev_layer)
    b = Conv2D(128//ds, (3, 3), strides=(1, 1), padding='same')(b)

    c = Conv2D(96//ds, (1, 1), strides=(1, 1),
               activation='relu', padding='same')(prev_layer)
    c = Conv2D(128//ds, (5, 5), strides=(1, 1), padding='same')(c)

    d = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(prev_layer)
    d = Conv2D(32//ds, (1, 1), strides=(1, 1), padding='same')(d)

    e = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(prev_layer)
    e = Conv2D(32//ds, (1, 1), strides=(1, 1), padding='same')(e)

    out_layer = concatenate([a, b, c, d, e], axis=-1)

    return out_layer


def inception_vae_2D(model_path, num_channels, dims, ds, learning_rate):

    intermediate_dim = 128 
    latent_dim = 2
    epsilon_std = 1.0

    ########## ENCODER ##########
    input_img = Input(shape=dims)

    x = inception_module_2D(input_img, ds=ds)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # residual inception modules
    for _ in range(3):
        x = inception_module_2D(x, ds=ds)
        y = Activation('relu')(x)
        x = inception_module_2D(x, ds=ds)
        x = add([x, y])
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

    flat = Flatten()(x)
    hidden = Dense(intermediate_dim, activation='relu')(flat)

    z_mean = Dense(latent_dim)(hidden)
    z_log_var = Dense(latent_dim)(hidden)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_var) * epsilon

    z = Lambda(sampling)([z_mean, z_log_var])
    ########## END ENCODER ##########

    ########## DECODER ##########
    latent_inputs = Input(shape=(latent_dim,), name="z_sampling")

    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    x = Dense(11 * 16 * (384//ds), activation='relu')(x)
    x = Reshape(target_shape=(11, 16, (384//ds)))(x)

    for _ in range(4):
        x = UpSampling2D((2, 2))(x)
        x = inception_module_2D(x, ds=ds)
        y = Activation('relu')(x)
        x = inception_module_2D(x, ds=ds)
        x = add([x, y])
        x = Activation('relu')(x)

    decoded = Conv2D(num_channels, (3, 3),
                     activation='linear', padding='same')(x)
    ########## END DECODER ##########

    # instantiate encoder/decoder
    encoder = Model(input_img, [z_mean, z_log_var, z], name='encoder')
    decoder = Model(latent_inputs, decoded, name='decoder')

    # here, the [2] index selects the 'z' tensor output as input to decoder
    # after passing inputs through the encoder
    output = decoder(encoder(input_img)[2])
    vae = Model(input_img, output)

    def mae_loss(y_true, y_pred, dims=dims):
        # need to scale up by number of pixels/voxels
        scaling = np.prod(dims)
        return scaling * metrics.mean_absolute_error(K.flatten(y_true),  K.flatten(y_pred))

    def kl_loss(y_true, y_pred):
        return -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

    def disentangled_kl_loss(y_true, y_pred):
        beta = BETA
        kl_loss = -0.5 * K.sum(1 + z_log_var -
                               K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return beta * kl_loss

    def vae_loss(y_true, y_pred):
        return K.mean(mae_loss(y_true, y_pred) + disentangled_kl_loss(y_true, y_pred))

    vae.compile(optimizer=Adam(lr=learning_rate),
                loss=vae_loss,
                metrics=[mae_loss, kl_loss, disentangled_kl_loss],)

    print(vae.summary())

    json_string = vae.to_json()
    with open(model_path, 'w') as f:
        json.dump(json_string, f)

    return encoder, decoder, vae


def vae_2D(model_path, num_channels, dims, ds, learning_rate):

    intermediate_dim = 256 
    latent_dim = 2
    epsilon_std = 1.0

    ########## ENCODER ##########
    input_img = Input(shape=dims)
    x = Conv2D(64//ds, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    for _ in range(3):
        x = Conv2D(64//ds, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

    flat = Flatten()(x)
    hidden = Dense(intermediate_dim, activation='relu')(flat)

    z_mean = Dense(latent_dim)(hidden)
    z_log_var = Dense(latent_dim)(hidden)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_var) * epsilon

    z = Lambda(sampling)([z_mean, z_log_var])
    ########## END ENCODER ##########

    ########## DECODER ##########
    latent_inputs = Input(shape=(latent_dim,), name="z_sampling")

    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    x = Dense(11*16*(64//ds), activation='relu')(x)
    x = Reshape(target_shape=(11, 16, (64//ds)))(x)

    for _ in range(4):
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64//ds, (3, 3), activation='relu', padding='same')(x)

    decoded = Conv2D(num_channels, (3, 3),
                     activation='linear', padding='same')(x)
    ########## END DECODER ##########

    # instantiate encoder/decoder
    encoder = Model(input_img, [z_mean, z_log_var, z], name='encoder')
    decoder = Model(latent_inputs, decoded, name='decoder')

    # here, the [2] index selects the 'z' tensor output as input to decoder
    # after passing inputs through the encoder
    output = decoder(encoder(input_img)[2])
    vae = Model(input_img, output)

    def mae_loss(y_true, y_pred, dims=dims):
        # need to scale up by number of pixels/voxels
        scaling = np.prod(dims)
        return scaling * metrics.mean_absolute_error(K.flatten(y_true),  K.flatten(y_pred))

    def kl_loss(y_true, y_pred):
        return -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

    def disentangled_kl_loss(y_true, y_pred):
        beta = BETA
        kl_loss = -0.5 * K.sum(1 + z_log_var -
                               K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return beta * kl_loss

    def vae_loss(y_true, y_pred):
        return K.mean(mae_loss(y_true, y_pred) + disentangled_kl_loss(y_true, y_pred))
        #return K.mean(correlation_coefficient_loss(y_true, y_pred) + disentangled_kl_loss(y_true, y_pred))

    vae.compile(optimizer=Adam(lr=learning_rate),
                loss=vae_loss,
                #metrics=[correlation_coefficient_loss, kl_loss, disentangled_kl_loss],)
                metrics=[mae_loss, kl_loss, disentangled_kl_loss],)

    print(vae.summary())

    json_string = vae.to_json()
    with open(model_path, 'w') as f:
        json.dump(json_string, f)

    return encoder, decoder, vae
