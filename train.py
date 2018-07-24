from utils.load_data import *
from utils.load_slice_data import *
from utils.display import *
from utils.utils import *
from models.vae import *

from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import time
import sys

if __name__ == '__main__':

    NUM_GPUS = 1

    ########## DIRECTORY SETUP ##########

    ROOT_DIR = "data"
    #TRAIN_DIR = os.path.join(ROOT_DIR, "train")
    TRAIN_DIR = os.path.join(ROOT_DIR, "train", "T1")
    cur_time = str(now())
    WEIGHT_DIR = os.path.join("models", "weights", cur_time)
    model_path = os.path.join(WEIGHT_DIR, "vae.json")

    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)

    ########## LOAD DATA ##########

    X, filenames, dims = load_slice_data(TRAIN_DIR, middle_only=False)

    ########## CALLBACKS ##########

    monitor_metric = "loss"
    # to train to convergence w/o running out of space
    checkpoint_filename = "most_recent_vae_weights.hdf5" 

    weight_path = os.path.join(WEIGHT_DIR, checkpoint_filename)

    mc = ModelCheckpoint(weight_path,
                         monitor='loss',
                         verbose=0,
                         save_best_only=False,
                         mode='auto',)
    es = EarlyStopping(monitor='loss',
                       min_delta=1e-6,
                       patience=100)
    callbacks_list = [mc, es]

    ########## MODEL SETUP ##########

    learning_rate = 1e-4
    encoder, decoder, model = inception_vae_2D(model_path=model_path,
                                     num_channels=X.shape[-1],
                                     ds=8,
                                     dims=dims,
                                     learning_rate=learning_rate,
                                     num_gpus=NUM_GPUS)

    ########## TRAIN ##########

    batch_size = 16 
    epochs = 10000000
    start_time = time.time()

    try:
        model.fit(X, X,
                  validation_split=0.2,
                  epochs=epochs,
                  batch_size=batch_size,
                  callbacks=callbacks_list,
                  verbose=1)
        print("Elapsed time: {:.4f}s".format(time.time() - start_time))

        plot_latent_sampling((encoder, decoder),
                             dims,
                             batch_size=batch_size)

    except:
        print("Elapsed time: {:.4f}s".format(time.time() - start_time))

        plot_latent_sampling((encoder, decoder),
                             dims,
                             batch_size=batch_size)

