import os

import numpy as np
import tensorflow as tf
import keras.layers as kl
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb




class Autoencoder():
    """ Small Convolutional Autoencoder"""

    def __init__(self, inp_dim=(128,96,3), state_dim=32, learning_rate=0.005):
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

        self.autoencoder.compile(optimizer="adam", loss="mse")
        self.autoencoder.summary()
        self.state_predicter.summary()

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

        x = Conv2DTranspose(12, (3, 3), activation='relu', padding='same')(x)
        x = Conv2DTranspose(24, (3, 3), activation='relu', strides=(1,2), padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        #x = kl.BatchNormalization()(x)
        x = Conv2DTranspose(24, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        #x = kl.BatchNormalization()(x)
        x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)
        autoencoder = Conv2DTranspose(8, (3, 3), strides=(2,2), activation='relu', padding='same')(x)

        return encoder, autoencoder

    def train(self, X, Y=None, num_episode=2000, batchsize=150):
        self.autoencoder.fit(X, X, batch_size=batchsize, epochs=num_episode)
        # IF you want to save the model
        model_json = self.autoencoder.to_json()
        with open("model_tex.json", "w") as json_file:
            json_file.write(model_json)

        self.autoencoder.save_weights("model_tex.h5")
        print("Saved model")


    def predict(self, X):

        result = self.autoencoder.predict(X, verbose=1, batch_size=100)
        result = self.decoder(X)


        return result

    def predict_states(self, X):
        states = self.state_predicter.predict(X, verbose=1, batch_size=100)
        return states
    def plot_images_autoencoder(self, orig_imgs, rec_imgs, episode):
        plt.close('all')
        plt.figure(figsize=(8, 2))
        print(np.shape(orig_imgs))
        print(np.shape(rec_imgs))
        for i in range(8):

            plt.subplot(2, 8, i+1)
            if(self.img_channel==3):
                plt.imshow(hsv_to_rgb(orig_imgs[i]))
            else:
                plt.imshow(orig_imgs[i,:,:,0], cmap='Greys',  interpolation='nearest')
            plt.axis('off')

            plt.subplot(2, 8, i+9)
            if(self.img_channel ==3):
                plt.imshow(hsv_to_rgb(rec_imgs[i]))
            else:
                plt.imshow(rec_imgs[i,:,:,0], cmap='Greys',  interpolation='nearest')


            plt.axis('off')
        plt.savefig("autoencodereps{}.png".format(episode))
        plt.pause(0.001)



def main():

    ae = Autoencoder()
    X = None# get_data()
    ae.train(X)
    result = ae.predict(X)
    img = result[0].reshape((128, 96, 3))
    plt.imshow(img)


main()
