import os

import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
import keras.layers as kl
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import datetime
from load_images import load_images


class Autoencoder():
    """ Small Convolutional Autoencoder"""

    def __init__(self, inp_dim=(128,96,3), state_dim=32, learning_rate=0.005, loss_weights=False):
        """
        inp_dim:  Dimension of Input feature
        out_dim:  Number of Classes
        learning_rate: Learning Rate for the Adam Optimizer
        """
        self.input_img = Input(shape=inp_dim)
        self. inp_dim= inp_dim
        self.state_dim = state_dim
        self.lr = learning_rate
        # Input and Output variable for training
        #self.x = tf.placeholder(tf.float32, shape=[None, self.inp_dim], name="inp_var")


        #input_shape = tf.shape(self.x)
        # Make Neural Net
        self.encoder, self.autoencoder = self._make_model()

        self.autoencoder = Model(self.input_img, self.autoencoder)

        self.state_predicter = Model(inputs=self.autoencoder.input, outputs=self.encoder)#self.autoencoder.get_layer("Encoder"))
        if loss_weights:
            #assert(np.shape(loss_weights) == inp_dim)
            #self.loss_weight = tf.constant(np.asarray(loss_weights), dtype=tf.float32)
            self.loss_weight = tf.keras.backend.placeholder(dtype=tf.float32, shape=inp_dim) #tuple([None] + inp_dim))

        else:
            self.loss_weight = tf.constant(np.ones(inp_dim), dtype=tf.float32)
        self.autoencoder.compile(optimizer="adam", loss=self.weighted_mse)
        self.autoencoder.summary()
        self.state_predicter.summary()

    def weighted_mse(self, y_true, y_pred):
        std = tf.math.reduce_std(y_true,axis=[0])
        print(tf.shape(std))
        return tf.reduce_mean(std * tf.square(y_true - y_pred))

    def _make_model(self, l1_reg=0.0001):
        """Creates a Neural Net"""
        regularizers_inp = {"w": tf.contrib.layers.l1_regularizer(scale=l1_reg),
                            "b": tf.contrib.layers.l1_regularizer(scale=l1_reg)}
        regularizers_mid = {"w": tf.contrib.layers.l1_regularizer(scale=l1_reg),
                            "b": tf.contrib.layers.l1_regularizer(scale=l1_reg)}
        regularizers_out = {"w": tf.contrib.layers.l1_regularizer(scale=l1_reg),
                            "b": tf.contrib.layers.l1_regularizer(scale=l1_reg)}


        x = Conv2D(8, (3, 3), strides=(2,2), activation='relu', padding='same')(self.input_img)
        #x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(24, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(24, (3, 3), activation='relu', strides=(1,2), padding='same')(x)
        #x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(12, (3, 3), activation='relu', padding='same')(x)


        x = Flatten()(x)
        encoder = Dense(self.state_dim, activation="softmax", name="Encoder")(x)


        # at this point the representation is (16, 6, 24) i.e. 2304-dimensional
        x = Dense(1152, activation='relu')(encoder)
        # DECODER
        x = kl.Reshape((16, 6, 12))(x) # TODO hardcoded dimensions

        x = Conv2DTranspose(24, (3, 3), activation='relu', padding='same')(x)
        x = Conv2DTranspose(24, (3, 3), activation='relu', strides=(1,2), padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        #x = kl.BatchNormalization()(x)
        x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        #x = kl.BatchNormalization()(x)
        x = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(x)
        autoencoder = Conv2DTranspose(3, (3, 3), strides=(2,2), activation='sigmoid', padding='same')(x)

        return encoder, autoencoder

    def train(self, X, Y=None, loss_weights=None, num_episode=300, batchsize=150):
        """
        self.autoencoder.fit(X, X, batch_size=batchsize, epochs=1)
        result = self.predict(X[:100])
        img = result[0].reshape((128, 96, 3))
        plt.imshow(img)
        plt.show()
        self.autoencoder.fit(X, X, batch_size=batchsize, epochs=1)
        result = self.predict(X[:100])
        img = result[0].reshape((128, 96, 3))
        plt.imshow(img)
        """
        checkpoint = ModelCheckpoint("logs/weights-improvement-{epoch:02d}.hdf5", verbose=1, period=10)
        logdir = "logs/" + datetime.time().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

        callbacks_list = [checkpoint, tensorboard_callback]

        # Fit the model
        self.loss_weight = loss_weights
        self.autoencoder.fit(X, X, batch_size=batchsize, epochs=num_episode, callbacks=callbacks_list)

        # IF you want to save the model
        model_json = self.autoencoder.to_json()
        with open("model_tex.json", "w") as json_file:
            json_file.write(model_json)

        self.autoencoder.save_weights("model_tex.h5")
        print("Saved model")


    def predict(self, X):
        result = self.autoencoder.predict(X, verbose=1, batch_size=100)
        return result

    def predict_states(self, X):
        states = self.state_predicter.predict(X, verbose=1, batch_size=100)
        return states

    def plot_images_autoencoder(self, orig_imgs, rec_imgs, episode=""):
        plt.close('all')
        plt.figure(figsize=(32, 8))
        print(np.shape(orig_imgs))
        print(np.shape(rec_imgs))
        for i in range(8):

            plt.subplot(2, 8, i+1)
            plt.imshow(orig_imgs[i])
            plt.axis('off')

            plt.subplot(2, 8, i+9)
            plt.imshow(rec_imgs[i])
            plt.axis('off')

        plt.savefig("autoencodereps{}.png".format(episode))
        plt.pause(0.001)

    def load(self, filename):
        self.autoencoder.load_weights(filename)


def main(train=False):

    ae = Autoencoder()

    X, Y = load_images("output.mp4", "label.csv", 128, 96)
    if (train):
        ae.train(X,batchsize=100)

    else:
        ae.load("logs/weights-improvement-100.hdf5")

    result = ae.predict(X[:100])
    img = result[0].reshape((128, 96, 3))
    plt.imshow(img)
    for i in range(5):
        ae.plot_images_autoencoder(X[i*10:(i+1)*10], result[i*10:(i+1)*10], i)

main(False)
